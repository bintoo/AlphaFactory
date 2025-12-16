from AlgorithmImports import *
import math
from datetime import timedelta

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Need long daily history for monthly return estimates
        self.SetWarmUp(timedelta(days=365))

        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # Use SPY as the "market" proxy for beta computation.
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

        # ACTIVE GUARD: always run exit/position management (reductions/closures) every bar
        # Use the most recently computed targets (may be empty before first rebalance)
        if not self.IsWarmingUp and self.targetWeights is not None:
            self._RebalanceToTargetWeights(data, self.targetWeights)

        # Monthly rebalance trigger: first data point in a new month
        if self.Time.month != self.lastRebalanceMonth:
            self.rebalancePending = True

        # Still allow maintenance/exit logic above to run, but only compute new targets when pending
        if not self.rebalancePending:
            return

        # Only compute new targets after warmup (maintenance/exit above still runs during warmup)
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

        if len(eligible) == 0:
            return

        metric = sigma if self.mode == "VolatilityQuintile" else beta

        selected = self._SelectLowestQuintile(eligible, metric)
        if len(selected) == 0:
            return

        # Weighting proxy: dollar volume (since true market cap not available in allowed API list)
        caps = {}
        capSum = 0.0
        for s in selected:
            cap = self._DollarVolumeProxy(s, 60, self.Time)
            if cap is None or cap <= 0:
                continue
            caps[s] = cap
            capSum += cap

        if capSum <= 0:
            return

        targetWeights = {}
        for s in selected:
            if s in caps:
                targetWeights[s] = caps[s] / capSum

        if len(targetWeights) == 0:
            return

        # Store targets so exit/position management can run on every OnData call
        self.targetWeights = targetWeights

        # Mark the rebalance as completed (orders will be managed by the continuous rebalance call)
        self.lastRebalanceMonth = self.Time.month
        self.rebalancePending = False

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
        # finalize month M only when we observe the first bar of month M+1.
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

        # Do NOT append lastClose here; that would include the current (possibly incomplete) month.

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

    def _SelectLowestQuintile(self, symbols: list, metric: dict):
        ranked = []
        for s in symbols:
            if s in metric and metric[s] is not None:
                ranked.append((s, float(metric[s])))

        if len(ranked) == 0:
            return []

        ranked.sort(key=lambda x: x[1])
        k = max(1, int(len(ranked) * 0.2))
        return [ranked[i][0] for i in range(k)]

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