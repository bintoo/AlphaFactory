from AlgorithmImports import *
import numpy as np

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Use Minute so exit logic can run every minute and we can avoid daily History(open) look-ahead ambiguity
        self.SetWarmUp(timedelta(days=90))

        # Universe (must use provided tickers)
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        self.symbols = []
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Minute)
            self.symbols.append(equity.Symbol)

        # -----------------------------
        # Paper parameters (exposed for sweep)
        # -----------------------------
        self.tauCandidates = [1, 2, 3, 4, 5, 6]
        self.etaPctCandidates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        self.paramIndexTau = 0
        self.paramIndexEta = 0

        self.tau = int(self.tauCandidates[self.paramIndexTau])
        self.etaPct = float(self.etaPctCandidates[self.paramIndexEta])

        self.transactionCostRate = 0.001

        # Control portfolio switch (paper's self-cause-only benchmark)
        self.useSelfCausalityControl = False

        # Training/test protocol: fixed 80/20 split date over the algorithm date range
        self.sampleStartDate = self.StartDate.date()
        self.sampleEndDate = self.EndDate.date()
        self.totalDays = (self.sampleEndDate - self.sampleStartDate).days
        self.splitDate = (self.sampleStartDate + timedelta(days=int(0.8 * self.totalDays)))

        # Data state (daily close series built from minute data)
        self.priceHistory = {s: [] for s in self.symbols}
        self.dates = []  # aligned daily timestamps (date) for history appends

        # Track daily close availability from minute data
        self.lastHistoryDate = None
        self.lastDayClose = {s: None for s in self.symbols}

        # Causal structure (discovered from training only, fixed during backtest)
        self.parentsBySymbol = {s: [] for s in self.symbols}
        self.causalGraphReady = False

        # Daily cycle state
        self.todayPredictedReturn = {}
        self.prevWinners = []
        self.prevLosers = []
        self.prevEntryPrice = {}
        self.prevTradeDate = None
        self.portfolioReturnSeries = []
        self.tradingDaysInTest = 0

        # Execution state: trade at next day's "open minute"; close by end-of-day
        self.pendingWinners = []
        self.pendingLosers = []
        self.pendingSignalDate = None

        # Intraday execution helpers
        self.openPriceToday = {s: None for s in self.symbols}
        self.openCapturedDate = None
        self.lastMinuteManagedDate = None

        self.eta = 1

        self.RecoverState()

    def RecoverState(self):
        self.todayPredictedReturn = {}
        self.prevWinners = []
        self.prevLosers = []
        self.prevEntryPrice = {}
        self.prevTradeDate = None
        self.portfolioReturnSeries = []
        self.tradingDaysInTest = 0

        self.pendingWinners = []
        self.pendingLosers = []
        self.pendingSignalDate = None

        self.lastHistoryDate = None
        self.lastDayClose = {s: None for s in self.symbols}

        self.openPriceToday = {s: None for s in self.symbols}
        self.openCapturedDate = None
        self.lastMinuteManagedDate = None

        for symbol in self.symbols:
            _ = self.Portfolio[symbol].Invested

    def OnData(self, data: Slice):
        day = self.Time.date()

        # 0) Update per-minute caches for "daily" series construction (runs during warmup too)
        for symbol in self.symbols:
            if data.Bars.ContainsKey(symbol):
                self.lastDayClose[symbol] = float(data.Bars[symbol].Close)

        # Capture today's "open" for each symbol from the first available minute bar (no History(open) usage)
        if self.openCapturedDate != day:
            captured_any = False
            for symbol in self.symbols:
                if data.Bars.ContainsKey(symbol) and self.openPriceToday.get(symbol, None) is None:
                    self.openPriceToday[symbol] = float(data.Bars[symbol].Open)
                    captured_any = True
            if captured_any:
                self.openCapturedDate = day

        # 1) Append ONE daily close per calendar day (safe, based on last seen minute close of the day)
        # Append the previous day's close when we detect a new day.
        if self.lastHistoryDate is None:
            self.lastHistoryDate = day
        elif day != self.lastHistoryDate:
            prev_day = self.lastHistoryDate

            # Only append if we have at least one close value for the previous day across the universe
            any_prev = False
            for symbol in self.symbols:
                if self.lastDayClose.get(symbol, None) is not None:
                    any_prev = True
                    break

            if any_prev:
                self.dates.append(prev_day)
                for symbol in self.symbols:
                    px = self.lastDayClose.get(symbol, None)
                    if px is None:
                        hist = self.priceHistory[symbol]
                        if len(hist) > 0:
                            hist.append(float(hist[-1]))
                        else:
                            hist.append(float("nan"))
                    else:
                        self.priceHistory[symbol].append(float(px))

            # Reset per-day state for new day
            self.lastHistoryDate = day
            self.openPriceToday = {s: None for s in self.symbols}
            self.openCapturedDate = None
            # Do not clear lastDayClose; it will be overwritten as new minute bars arrive.

        # 2) Minute-frequency management guard (must run exit logic every minute, not daily-gated)
        if not self.IsWarmingUp:
            self.ManageIntradayExecutionAndExit(data)

        # 3) After daily close series updated and after split, run causal discovery once
        # We run this during the test start day or later once enough training data exists.
        if (not self.causalGraphReady) and (day >= self.splitDate) and self.HasSufficientTrainingHistory():
            self.DiscoverCausalParentsFromTraining()
            self.causalGraphReady = True

        # 4) Signal generation at "end of day" based on our constructed daily closes.
        # We trigger once per day near the end (after close), but using Minute data.
        # If market hours differ, this approximation still stays causal because it only uses already-appended closes.
        if self.IsWarmingUp:
            return
        if day < self.splitDate:
            return
        if not self.causalGraphReady:
            return

        # Use a once-per-day trigger at/after 15:59 so we don't spam signals each minute.
        if self.lastMinuteManagedDate == day:
            # lastMinuteManagedDate is used for execution/exit; do not block signal generation.
            pass

        if self.Time.hour == 15 and self.Time.minute >= 59:
            # Prevent repeated signal generation in the last minutes
            if getattr(self, "lastSignalDate", None) != day:
                self.lastSignalDate = day
                self.GenerateSignalsAtClose()

    # -----------------------------
    # Intraday execution and exit (runs every minute)
    # -----------------------------
    def ManageIntradayExecutionAndExit(self, data: Slice):
        day = self.Time.date()

        # A) Execute pending trades at/near today's open minute once per day
        # We require open prices captured for all traded symbols.
        if self.pendingSignalDate is not None and self.pendingSignalDate != day:
            if getattr(self, "executedForDay", None) != day:
                self.ExecutePendingTradesAtOpen(data)
                if self.prevTradeDate == day:
                    self.executedForDay = day

        # B) Close positions near end of day (minute-frequency check)
        # Close during the last minute(s) of the session to approximate "end-of-day close out".
        # Use 15:59 as approximation.
        if self.prevTradeDate == day:
            if self.Time.hour == 15 and self.Time.minute >= 59:
                if getattr(self, "closedForDay", None) != day:
                    self.ClosePositionsForDayAndComputeReturn(data)
                    self.closedForDay = day

        self.lastMinuteManagedDate = day

    # -----------------------------
    # Training/test split helpers
    # -----------------------------
    def HasSufficientTrainingHistory(self) -> bool:
        train_len = self.GetTrainingLength()
        if train_len is None:
            return False
        return train_len >= max(60, 10 * int(self.tau))

    def GetTrainingLength(self):
        if len(self.dates) == 0:
            return None
        n = 0
        for d in self.dates:
            if d < self.splitDate:
                n += 1
        return n

    # -----------------------------
    # Causal discovery (VAR proxy, training-only)
    # -----------------------------
    def DiscoverCausalParentsFromTraining(self):
        train_len = self.GetTrainingLength()
        if train_len is None:
            return

        n_assets = len(self.symbols)
        Y = np.zeros((train_len, n_assets), dtype=float)

        for j, sym in enumerate(self.symbols):
            series = self.priceHistory[sym][:train_len]
            filled = []
            last = None
            for v in series:
                if v != v:
                    if last is None:
                        filled.append(np.nan)
                    else:
                        filled.append(last)
                else:
                    last = float(v)
                    filled.append(last)

            first_valid = None
            for v in filled:
                if v == v:
                    first_valid = v
                    break
            if first_valid is None:
                Y[:, j] = 0.0
            else:
                out = []
                carry = first_valid
                for v in filled:
                    if v == v:
                        carry = v
                        out.append(v)
                    else:
                        out.append(carry)
                Y[:, j] = np.asarray(out, dtype=float)

        tau = int(self.tau)
        if train_len <= tau + 5:
            return

        rows = train_len - tau
        X = np.zeros((rows, n_assets * tau + 1), dtype=float)
        X[:, 0] = 1.0

        for r in range(rows):
            t = tau + r
            col = 1
            for lag in range(1, tau + 1):
                X[r, col:col + n_assets] = Y[t - lag, :]
                col += n_assets

        parents = {sym: [] for sym in self.symbols}
        K = 3

        XtX = X.T.dot(X)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except Exception:
            return
        Xt = X.T

        for target_idx, target_sym in enumerate(self.symbols):
            y = Y[tau:, target_idx].reshape(-1, 1)
            beta = XtX_inv.dot(Xt).dot(y)

            influence = np.zeros(n_assets, dtype=float)
            for lag in range(1, tau + 1):
                start = 1 + (lag - 1) * n_assets
                end = start + n_assets
                b = beta[start:end, 0]
                influence += np.abs(b)

            cand = []
            for p_idx, p_sym in enumerate(self.symbols):
                if (not self.useSelfCausalityControl) and (p_sym == target_sym):
                    continue
                cand.append((float(influence[p_idx]), p_sym))

            cand.sort(key=lambda x: x[0])

            chosen = [s for (score, s) in cand[-K:] if score > 0.0]
            if self.useSelfCausalityControl:
                chosen = [target_sym]

            parents[target_sym] = chosen

        self.parentsBySymbol = parents

    # -----------------------------
    # Signal generation (expanding window regression per stock)
    # -----------------------------
    def GenerateSignalsAtClose(self):
        n = len(self.symbols)
        eta = int(max(1, round(self.etaPct * n)))
        eta = int(min(eta, max(1, n // 2)))
        self.eta = eta

        self.todayPredictedReturn = {}

        for x in self.symbols:
            gamma = self.PredictNextDayReturn_Expanding(x)
            if gamma is not None:
                self.todayPredictedReturn[x] = gamma

        eligible = list(self.todayPredictedReturn.keys())
        if len(eligible) < 2 * self.eta:
            self.pendingWinners = []
            self.pendingLosers = []
            self.pendingSignalDate = self.Time.date()
            return

        eligible_sorted = sorted(eligible, key=lambda s: self.todayPredictedReturn[s])
        losers = eligible_sorted[:self.eta]
        winners = eligible_sorted[-self.eta:]

        self.pendingWinners = winners
        self.pendingLosers = losers
        self.pendingSignalDate = self.Time.date()

    def PredictNextDayReturn_Expanding(self, x: Symbol):
        parents = self.parentsBySymbol.get(x, None)
        if parents is None or len(parents) == 0:
            return None

        x_hist = self.priceHistory.get(x, None)
        if x_hist is None:
            return None

        T = len(x_hist)
        if T < (int(self.tau) + 5):
            return None

        parent_hists = []
        for p in parents:
            h = self.priceHistory.get(p, None)
            if h is None or len(h) != T:
                return None
            parent_hists.append(h)

        min_samples = 30
        X_rows = []
        y = []

        for s in range(int(self.tau), T):
            if x_hist[s] != x_hist[s]:
                continue

            feats = []
            ok = True
            for h in parent_hists:
                for lag in range(1, int(self.tau) + 1):
                    idx = s - lag
                    v = h[idx]
                    if idx < 0 or v != v:
                        ok = False
                        break
                    feats.append(float(v))
                if not ok:
                    break

            if not ok:
                continue

            X_rows.append(feats)
            y.append(float(x_hist[s]))

        if len(y) < min_samples:
            return None

        Xmat = np.asarray(X_rows, dtype=float)
        yvec = np.asarray(y, dtype=float).reshape(-1, 1)

        ones = np.ones((Xmat.shape[0], 1), dtype=float)
        Xdesign = np.hstack([ones, Xmat])

        XtX = Xdesign.T.dot(Xdesign)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except Exception:
            return None

        beta = XtX_inv.dot(Xdesign.T).dot(yvec)

        t = T - 1
        pred_feats = []
        for h in parent_hists:
            for lag in range(0, int(self.tau)):
                idx = t - lag
                v = h[idx]
                if idx < 0 or v != v:
                    return None
                pred_feats.append(float(v))

        xpred = np.asarray([1.0] + pred_feats, dtype=float).reshape(1, -1)
        rho_next = float(xpred.dot(beta)[0, 0])

        P_t = float(x_hist[t])
        if P_t <= 0 or P_t != P_t:
            return None

        gamma = (rho_next - P_t) / P_t
        return float(gamma)

    # -----------------------------
    # Execution and daily close-out
    # -----------------------------
    def ExecutePendingTradesAtOpen(self, data: Slice):
        if self.pendingSignalDate is None:
            return

        # Execute only on a later date than the signal date (t -> t+1)
        if self.pendingSignalDate == self.Time.date():
            return

        winners = list(self.pendingWinners) if self.pendingWinners is not None else []
        losers = list(self.pendingLosers) if self.pendingLosers is not None else []

        if len(winners) != int(self.eta) or len(losers) != int(self.eta):
            return

        # Flatten first (paper: close out before next trading action)
        self.FlattenAllPositionsWithLimitOrders(data)

        # Entry price from captured minute open (causal, no History)
        entry_prices = {}
        for symbol in set(winners + losers):
            px = self.openPriceToday.get(symbol, None)
            if px is None or px <= 0:
                return
            entry_prices[symbol] = float(px)

        portfolio_value = float(self.Portfolio.TotalPortfolioValue)

        grossDollar = portfolio_value
        longGross = grossDollar / 2.0
        shortGross = grossDollar / 2.0
        targetLongDollarPerStock = longGross / float(self.eta)
        targetShortDollarPerStock = shortGross / float(self.eta)

        for symbol in winners:
            price = float(entry_prices[symbol])
            target_qty = int(targetLongDollarPerStock / price)
            if target_qty <= 0:
                continue

            current_qty = int(self.Portfolio[symbol].Quantity)
            delta = target_qty - current_qty
            if delta != 0:
                self.LimitOrder(symbol, delta, price)

        for symbol in losers:
            price = float(entry_prices[symbol])
            target_qty = -int(targetShortDollarPerStock / price)
            if target_qty >= 0:
                continue

            current_qty = int(self.Portfolio[symbol].Quantity)
            delta = target_qty - current_qty
            if delta != 0:
                self.LimitOrder(symbol, delta, price)

        self.prevWinners = winners
        self.prevLosers = losers
        self.prevEntryPrice = entry_prices
        self.prevTradeDate = self.Time.date()

        self.pendingWinners = []
        self.pendingLosers = []
        self.pendingSignalDate = None

    def ClosePositionsForDayAndComputeReturn(self, data: Slice):
        if self.prevTradeDate is None:
            return

        if self.prevTradeDate != self.Time.date():
            self.FlattenAllPositionsWithLimitOrders(data)
            self.prevTradeDate = None
            self.prevWinners = []
            self.prevLosers = []
            self.prevEntryPrice = {}
            return

        if len(self.prevWinners) != int(self.eta) or len(self.prevLosers) != int(self.eta):
            self.FlattenAllPositionsWithLimitOrders(data)
            self.prevTradeDate = None
            self.prevWinners = []
            self.prevLosers = []
            self.prevEntryPrice = {}
            return

        for s in self.prevWinners + self.prevLosers:
            if not data.Bars.ContainsKey(s):
                return
            if s not in self.prevEntryPrice:
                return

        long_rets = []
        for s in self.prevWinners:
            p0 = float(self.prevEntryPrice[s])
            p1 = float(data.Bars[s].Close)
            if p0 <= 0:
                return
            long_rets.append((p1 - p0) / p0)

        short_rets = []
        for s in self.prevLosers:
            p0 = float(self.prevEntryPrice[s])
            p1 = float(data.Bars[s].Close)
            if p0 <= 0:
                return
            short_rets.append((p1 - p0) / p0)

        avg_long = float(np.mean(long_rets)) if len(long_rets) > 0 else 0.0
        avg_short = float(np.mean(short_rets)) if len(short_rets) > 0 else 0.0

        realized = avg_long - avg_short
        realized -= float(self.transactionCostRate)

        if self.Time.date() >= self.splitDate:
            self.portfolioReturnSeries.append(float(realized))
            self.tradingDaysInTest += 1

        self.FlattenAllPositionsWithLimitOrders(data)

        self.prevTradeDate = None
        self.prevWinners = []
        self.prevLosers = []
        self.prevEntryPrice = {}

    def FlattenAllPositionsWithLimitOrders(self, data: Slice):
        for symbol in self.symbols:
            qty = int(self.Portfolio[symbol].Quantity)
            if qty == 0:
                continue
            if not data.Bars.ContainsKey(symbol):
                continue
            px = float(data.Bars[symbol].Close)
            if px <= 0:
                continue
            self.LimitOrder(symbol, -qty, px)

    def OnEndOfAlgorithm(self):
        Ttest = int(self.tradingDaysInTest)
        if Ttest <= 0 or len(self.portfolioReturnSeries) == 0:
            return

        cumulative = 1.0
        for r in self.portfolioReturnSeries:
            cumulative *= (1.0 + float(r))
        r_test = cumulative - 1.0

        annualized = (1.0 + r_test) ** (252.0 / float(Ttest)) - 1.0

        self.Log("TEST WINDOW: splitDate=%s  days=%d  cumulative=%.6f  annualized=%.6f" %
                 (str(self.splitDate), Ttest, float(r_test), float(annualized)))

        self.Log("PARAMS: tau=%d etaPct=%.3f eta=%d selfControl=%s" %
                 (int(self.tau), float(self.etaPct), int(self.eta), str(self.useSelfCausalityControl)))