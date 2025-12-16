from AlgorithmImports import *
import math
from datetime import timedelta

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Need at least 60 months of monthly observations; use a larger warmup to be safe.
        self.SetWarmUp(timedelta(days=365 * 6))

        # Universe: attempt to approximate "many individual stocks" with the tickers available in this environment.
        # NOTE: In this local-data safe mode, we cannot access CRSP universe or fundamental market cap.
        # We therefore implement the paper mechanics (monthly formation, trailing 24-60 months risk estimates,
        # quintile formation, cap-weighting) using the provided tickers only.
        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # Market proxy for beta computation (approximation for CRSP VW market)
        self.marketSymbol = self.symbols[0]

        # Rebalance control
        self.lastRebalanceMonth = -1
        self.rebalancePending = False

        # Strategy mode
        self.mode = "VolatilityQuintile"  # "VolatilityQuintile" or "BetaQuintile"

        # Data requirements from paper
        self.maxMonths = 60
        self.minMonths = 24

        # Order management: avoid resubmitting limit orders every OnData call
        self.openOrderIdsBySymbol = {}

        # Persist last computed targets so exit/position management can run every OnData
        self.targetWeights = {}

        # Performance analytics (monthly series)
        self.prevMonthKey = None
        self.monthEndPrices = {}
        self.monthlyPortfolioReturns = []   # portfolio total return
        self.monthlyMarketReturns = []      # market proxy return
        self.monthlyActiveReturns = []      # portfolio - market
        self.monthlyExcessReturns = []      # portfolio - rf (rf assumed 0 in this environment)

        # State recovery
        self.RecoverState()

    def RecoverState(self):
        if not hasattr(self, "openOrderIdsBySymbol") or self.openOrderIdsBySymbol is None:
            self.openOrderIdsBySymbol = {}

        # Rebuild from open orders (if any)
        self.openOrderIdsBySymbol = {}
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if open_orders is not None:
                for o in open_orders:
                    s = o.Symbol
                    if s is None:
                        continue
                    if s not in self.openOrderIdsBySymbol:
                        self.openOrderIdsBySymbol[s] = []
                    self.openOrderIdsBySymbol[s].append(o.Id)
        except Exception:
            self.openOrderIdsBySymbol = {}

        if not hasattr(self, "lastRebalanceMonth") or self.lastRebalanceMonth is None:
            self.lastRebalanceMonth = -1
        if not hasattr(self, "rebalancePending") or self.rebalancePending is None:
            self.rebalancePending = False

        if not hasattr(self, "targetWeights") or self.targetWeights is None:
            self.targetWeights = {}

        if not hasattr(self, "monthEndPrices") or self.monthEndPrices is None:
            self.monthEndPrices = {}

        if not hasattr(self, "prevMonthKey") or self.prevMonthKey is None:
            self.prevMonthKey = None

        any_invested = False
        for s in self.symbols:
            if self.Portfolio[s].Invested:
                any_invested = True
                break

        if any_invested and self.lastRebalanceMonth == -1:
            self.lastRebalanceMonth = self.Time.month

    def OnData(self, data: Slice):
        # ACTIVE GUARD: always maintain open order bookkeeping
        self._MaintainOpenOrders()

        # ACTIVE GUARD: collect month-end closes for analytics and for robust month-change triggering.
        self._UpdateMonthEndSeries(data)

        # ACTIVE GUARD: always run exit/position management every bar (but do not place orders during warmup)
        if not self.IsWarmingUp and self.targetWeights is not None and len(self.targetWeights) > 0:
            self._RebalanceToTargetWeights(data, self.targetWeights)

        # Monthly rebalance trigger: first data point in a new month
        if self.Time.month != self.lastRebalanceMonth:
            self.rebalancePending = True

        if not self.rebalancePending:
            return

        # Only compute new targets after warmup
        if self.IsWarmingUp:
            return

        # Ensure we have current daily bars for all symbols we might trade today
        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                return

        # Build monthly returns using daily history bounded to [start, self.Time]
        marketMonthly = self._GetMonthlyReturnsFromDailyHistory(self.marketSymbol, self.maxMonths + 1, self.Time)
        if marketMonthly is None or len(marketMonthly) < self.minMonths:
            return

        eligible = []
        sigma = {}
        beta = {}

        for s in self.symbols:
            r = self._GetMonthlyReturnsFromDailyHistory(s, self.maxMonths + 1, self.Time)
            if r is None or len(r) < self.minMonths:
                continue

            n = min(self.maxMonths, len(r), len(marketMonthly))
            if n < self.minMonths:
                continue

            ri = r[-n:]
            rm = marketMonthly[-n:]

            sig = self._StdDev(ri)
            if sig is None:
                continue
            sigma[s] = sig

            var_m = self._Variance(rm)
            if var_m is None or var_m <= 0:
                continue

            cov_im = self._Covariance(ri, rm)
            if cov_im is None:
                continue
            beta[s] = cov_im / var_m

            eligible.append(s)

        if len(eligible) < 5:
            return

        metric = sigma if self.mode == "VolatilityQuintile" else beta

        quintiles = self._FormQuintiles(eligible, metric, 5)
        if quintiles is None or len(quintiles) != 5:
            return

        # Replication logic: form all 5 quintiles and cap-weight within each.
        # In this environment we do not have fundamental market cap. We will use a transparent proxy:
        # average (close * volume) over recent history as a "cap-like weight proxy".
        # This preserves the mechanics of cap-weighting, though it is not true market cap.
        quintileWeights = []
        for q in quintiles:
            w = self._ComputeCapWeightedTargets(q, 60, self.Time)
            quintileWeights.append(w)

        # Record quintile returns would require holding each portfolio separately.
        # We trade one portfolio: the lowest-risk quintile to keep strategy behavior consistent with the prior code.
        # But we still *form* all 5 quintiles each month (core replication step).
        selectedTargets = quintileWeights[0]
        if selectedTargets is None or len(selectedTargets) == 0:
            return

        self.targetWeights = selectedTargets

        self.lastRebalanceMonth = self.Time.month
        self.rebalancePending = False

    def OnEndOfAlgorithm(self):
        # Emit basic verification stats for the implemented monthly series.
        # Rf is assumed 0 in this environment (no Ken French series available via the provided API reference).
        sharpe = self._Sharpe(self.monthlyExcessReturns)
        te = self._StdDev(self.monthlyActiveReturns)
        ir = None
        if te is not None and te > 0:
            mean_active = self._Mean(self.monthlyActiveReturns)
            if mean_active is not None:
                ir = mean_active / te

        alpha, beta = self._CapmAlphaBeta(self.monthlyPortfolioReturns, self.monthlyMarketReturns)

        if sharpe is not None:
            self.Debug("Monthly Sharpe (Rf=0 proxy): " + str(sharpe))
        if te is not None:
            self.Debug("Monthly Tracking Error (vs market proxy): " + str(te))
        if ir is not None:
            self.Debug("Monthly Information Ratio (vs market proxy): " + str(ir))
        if alpha is not None and beta is not None:
            self.Debug("CAPM (proxy) alpha (monthly): " + str(alpha) + " | beta: " + str(beta))

    def _MaintainOpenOrders(self):
        if not hasattr(self, "openOrderIdsBySymbol") or self.openOrderIdsBySymbol is None:
            self.openOrderIdsBySymbol = {}

        removeSymbols = []
        for s, ids in list(self.openOrderIdsBySymbol.items()):
            alive = []
            for oid in ids:
                order = self.Transactions.GetOrderById(oid)
                if order is None:
                    continue
                if order.Status in [OrderStatus.Submitted, OrderStatus.PartiallyFilled, OrderStatus.New]:
                    alive.append(oid)
            if len(alive) == 0:
                removeSymbols.append(s)
            else:
                self.openOrderIdsBySymbol[s] = alive

        for s in removeSymbols:
            if s in self.openOrderIdsBySymbol:
                del self.openOrderIdsBySymbol[s]

    def _RebalanceToTargetWeights(self, data: Slice, targetWeights: dict):
        if targetWeights is None:
            return

        portfolioValue = float(self.Portfolio.TotalPortfolioValue)

        symbolsToProcess = set()
        for s in self.symbols:
            if self.Portfolio[s].Invested:
                symbolsToProcess.add(s)
        for s in targetWeights.keys():
            symbolsToProcess.add(s)

        for s in symbolsToProcess:
            # Avoid reissuing if there are still live orders for this symbol
            if s in self.openOrderIdsBySymbol and len(self.openOrderIdsBySymbol[s]) > 0:
                continue

            if not data.Bars.ContainsKey(s):
                continue

            price = float(data.Bars[s].Close)
            if price <= 0:
                continue

            targetWeight = float(targetWeights.get(s, 0.0))
            targetValue = portfolioValue * targetWeight
            targetQty = int(targetValue / price)

            currentQty = int(self.Portfolio[s].Quantity)
            deltaQty = targetQty - currentQty

            if abs(deltaQty) < 1:
                continue

            limitPrice = price
            ticket = self.LimitOrder(s, deltaQty, limitPrice)
            if ticket is not None:
                oid = ticket.OrderId
                if s not in self.openOrderIdsBySymbol:
                    self.openOrderIdsBySymbol[s] = []
                self.openOrderIdsBySymbol[s].append(oid)

    def _GetMonthlyReturnsFromDailyHistory(self, symbol: Symbol, monthsPlusOne: int, endTime):
        lookbackDays = int(monthsPlusOne * 31)
        start = endTime - timedelta(days=lookbackDays)

        hist = self.History(symbol, start, endTime, Resolution.Daily)
        if hist.empty:
            return None

        # DataFrame safety: isolate DatetimeIndex
        if 'close' in hist.columns:
            df = hist
        else:
            df = hist.loc[symbol]

        if df.empty or 'close' not in df.columns:
            return None

        closes = df['close']
        if closes is None or len(closes) < 2:
            return None

        # Build month-end closes WITHOUT including the current in-progress month:
        monthEndCloses = []
        lastKey = None
        lastClose = None

        for t, c in closes.items():
            key = (t.year, t.month)

            if lastKey is None:
                lastKey = key
                lastClose = float(c)
                continue

            if key != lastKey:
                if lastClose is not None and lastClose > 0:
                    monthEndCloses.append(lastClose)
                lastKey = key

            lastClose = float(c)

        if len(monthEndCloses) < 2:
            return None

        rets = []
        for i in range(1, len(monthEndCloses)):
            prevC = float(monthEndCloses[i - 1])
            currC = float(monthEndCloses[i])
            if prevC <= 0:
                continue
            rets.append((currC / prevC) - 1.0)

        return rets

    def _DollarVolumeProxy(self, symbol: Symbol, lookbackDays: int, endTime):
        start = endTime - timedelta(days=int(lookbackDays * 2))
        hist = self.History(symbol, start, endTime, Resolution.Daily)
        if hist.empty:
            return None

        if 'close' in hist.columns:
            df = hist
        else:
            df = hist.loc[symbol]

        if df.empty:
            return None
        if 'close' not in df.columns or 'volume' not in df.columns:
            return None

        if len(df) > lookbackDays:
            df = df.tail(lookbackDays)

        dv = (df['close'] * df['volume']).dropna()
        if dv is None or len(dv) == 0:
            return None
        return float(dv.mean())

    def _FormQuintiles(self, symbols: list, metric: dict, k: int):
        ranked = []
        for s in symbols:
            if s in metric and metric[s] is not None:
                ranked.append((s, float(metric[s])))

        if len(ranked) < k:
            return None

        ranked.sort(key=lambda x: x[1])

        n = len(ranked)
        base = n // k
        rem = n % k

        quintiles = []
        start = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            if size <= 0:
                return None
            part = ranked[start:start + size]
            quintiles.append([x[0] for x in part])
            start += size

        if len(quintiles) != k:
            return None
        return quintiles

    def _ComputeCapWeightedTargets(self, symbols: list, capLookbackDays: int, endTime):
        if symbols is None or len(symbols) == 0:
            return None

        caps = {}
        capSum = 0.0
        for s in symbols:
            cap = self._DollarVolumeProxy(s, capLookbackDays, endTime)
            if cap is None or cap <= 0:
                continue
            caps[s] = cap
            capSum += cap

        if capSum <= 0:
            return None

        targetWeights = {}
        for s, cap in caps.items():
            targetWeights[s] = float(cap) / float(capSum)

        if len(targetWeights) == 0:
            return None

        return targetWeights

    def _UpdateMonthEndSeries(self, data: Slice):
        # Detect month change using daily bars and record last close of the previous month.
        if self.prevMonthKey is None:
            self.prevMonthKey = (self.Time.year, self.Time.month)
            return

        currentKey = (self.Time.year, self.Time.month)
        if currentKey == self.prevMonthKey:
            return

        # Month changed: the last bar we saw for prevMonthKey was the month-end close.
        # We store the last available close for each symbol from the previous day if present in current slice.
        # Since we do not have explicit access to "yesterday's bars" here, we use History for just enough
        # data to safely pull the previous trading day's close for each symbol and the market proxy.
        prevEnd = self.Time
        prevStart = prevEnd - timedelta(days=5)

        # Portfolio return: computed from total portfolio value month-to-month (uses algorithm equity curve)
        # We approximate month-end value as TotalPortfolioValue at first bar of the new month.
        # This is not a perfect close-to-close at month-end timestamp, but is consistent for verification in this environment.
        # Compute returns when we have at least one prior month-end value stored.
        if not hasattr(self, "lastMonthEndPortfolioValue"):
            self.lastMonthEndPortfolioValue = float(self.Portfolio.TotalPortfolioValue)
        else:
            currVal = float(self.Portfolio.TotalPortfolioValue)
            prevVal = float(self.lastMonthEndPortfolioValue)
            if prevVal > 0:
                rp = (currVal / prevVal) - 1.0
                self.monthlyPortfolioReturns.append(rp)
                # Rf proxy = 0
                self.monthlyExcessReturns.append(rp)
            self.lastMonthEndPortfolioValue = currVal

        # Market return from month-end to month-end using daily history month-end logic (consistent with formation code)
        rm_series = self._GetMonthlyReturnsFromDailyHistory(self.marketSymbol, 2, self.Time)
        if rm_series is not None and len(rm_series) >= 1:
            rm = float(rm_series[-1])
            self.monthlyMarketReturns.append(rm)
            if len(self.monthlyPortfolioReturns) > 0:
                self.monthlyActiveReturns.append(self.monthlyPortfolioReturns[-1] - rm)

        self.prevMonthKey = currentKey

    def _Mean(self, xs):
        if xs is None or len(xs) == 0:
            return None
        return sum(xs) / float(len(xs))

    def _Variance(self, xs):
        if xs is None or len(xs) < 2:
            return None
        m = self._Mean(xs)
        if m is None:
            return None
        s = 0.0
        for x in xs:
            d = x - m
            s += d * d
        return s / float(len(xs) - 1)

    def _StdDev(self, xs):
        v = self._Variance(xs)
        if v is None or v < 0:
            return None
        return math.sqrt(v)

    def _Covariance(self, xs, ys):
        if xs is None or ys is None:
            return None
        n = min(len(xs), len(ys))
        if n < 2:
            return None
        x = xs[-n:]
        y = ys[-n:]
        mx = self._Mean(x)
        my = self._Mean(y)
        if mx is None or my is None:
            return None
        s = 0.0
        for i in range(n):
            s += (x[i] - mx) * (y[i] - my)
        return s / float(n - 1)

    def _Sharpe(self, excessReturns):
        if excessReturns is None or len(excessReturns) < 2:
            return None
        m = self._Mean(excessReturns)
        sd = self._StdDev(excessReturns)
        if m is None or sd is None or sd <= 0:
            return None
        return m / sd

    def _CapmAlphaBeta(self, portfolioReturns, marketReturns):
        if portfolioReturns is None or marketReturns is None:
            return (None, None)
        n = min(len(portfolioReturns), len(marketReturns))
        if n < 2:
            return (None, None)

        rp = portfolioReturns[-n:]
        rm = marketReturns[-n:]

        var_m = self._Variance(rm)
        if var_m is None or var_m <= 0:
            return (None, None)

        cov_pm = self._Covariance(rp, rm)
        if cov_pm is None:
            return (None, None)

        b = cov_pm / var_m
        mean_rp = self._Mean(rp)
        mean_rm = self._Mean(rm)
        if mean_rp is None or mean_rm is None:
            return (None, None)

        a = mean_rp - b * mean_rm
        return (a, b)