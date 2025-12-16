from AlgorithmImports import *
import numpy as np

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Paper workflow is daily, end-of-day rebalance
        self.SetWarmUp(timedelta(days=30))

        self.tau = 2
        self.eta = 2
        self.transactionCostRate = 0.001  # fixed daily cost C in Eq.(3), accounting-only

        # Use daily resolution to align with "end-of-day" close-based workflow
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        self.symbols = []
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # Train/test split parameters (80/20) per paper
        self.trainingFraction = 0.8
        self.trainEndIndex = None
        self.trainEndDate = None

        # Bookkeeping for close series (daily bars -> just append closes)
        self.dailyCloseHistory = {s: [] for s in self.symbols}
        self.dailyDateHistory = {s: [] for s in self.symbols}

        # Track last processed daily date to ensure "once per day" logic
        self.lastProcessedDate = None

        # Track holdings (1-day holding period, liquidate daily)
        self.currentLongs = set()
        self.currentShorts = set()
        self.targetLongs = set()
        self.targetShorts = set()

        # Transaction cost accounting per Eq.(3)
        self.dailyRp = []          # realized daily portfolio return series (after cost)
        self.lastEquityValue = None

        # Control portfolio flag: self-cause-only (PA_X = {X})
        # Paper compares causal discovery portfolio vs self-cause-only control.
        # Default: causal discovery portfolio; set True to run control.
        self.useSelfCauseOnly = False

        # Causal discovery algorithm configuration (placeholder-friendly structure)
        # We implement a sparse, data-driven lagged-parent selection on TRAIN ONLY to avoid
        # the "fully-connected graph" invalidation. This serves as the causal discovery phase.
        self.causalAlgorithm = "LaggedCorrelation"  # stand-in for tsFCI/VarLiNGAM/TiMINo runs
        self.maxParents = 3

        self.parents = {s: [] for s in self.symbols}
        self.parentsComputed = False

        self.RecoverState()

    def RecoverState(self):
        self.currentLongs = set()
        self.currentShorts = set()

        for s in self.symbols:
            p = self.Portfolio[s]
            if not p.Invested:
                continue
            if p.Quantity > 0:
                self.currentLongs.add(s)
            elif p.Quantity < 0:
                self.currentShorts.add(s)

        self.targetLongs = set(self.currentLongs)
        self.targetShorts = set(self.currentShorts)

        self.lastProcessedDate = None
        self.lastEquityValue = None

    def OnData(self, data: Slice):
        # Update close history (even during warmup; no trading until ready)
        self._UpdateDailyCloseHistory(data)

        # Need a bar for each symbol to consider the day's close complete
        current_date = self.Time.date()
        if self.lastProcessedDate == current_date:
            return

        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                return

        # Mark this date as processed (we're at daily bar time)
        self.lastProcessedDate = current_date

        # Compute train/test split once we have enough daily history
        if self.trainEndIndex is None:
            self._TryComputeTrainTestSplit()
            if self.trainEndIndex is None:
                return

        # Causal discovery phase (TRAIN ONLY) -> parents[X] = PA_X
        if not self.parentsComputed:
            self._TryRunCausalDiscoveryOnTraining()
            if not self.parentsComputed:
                return

        # Do not trade during warmup, but keep collecting data
        if self.IsWarmingUp:
            return

        # Must have passed train end date to start test/backtest trading per paper
        if self.trainEndDate is None or current_date <= self.trainEndDate:
            return

        # End-of-day workflow:
        # 1) Close out previous day's positions at today's close
        self._CloseOutAllPositionsMarket()

        # 2) Compute predictions for next day using expanding window starting at training end
        predicted = self._ComputePredictedReturnsExpandingWindow()
        if len(predicted) < 2 * self.eta:
            self.targetLongs = set()
            self.targetShorts = set()
            self._AccountDailyReturnWithCost()
            return

        ranked = sorted(predicted.items(), key=lambda kv: kv[1], reverse=True)
        winners = [kv[0] for kv in ranked[:self.eta]]
        losers = [kv[0] for kv in ranked[-self.eta:]]

        self.targetLongs = set(winners)
        self.targetShorts = set(losers)

        # 3) Enter equal-weight long/short baskets at close (daily bar time -> MarketOrder here)
        total_value = float(self.Portfolio.TotalPortfolioValue)

        total_long_notional = 0.5 * total_value
        total_short_notional = 0.5 * total_value

        long_notional_per = total_long_notional / float(self.eta)
        short_notional_per = total_short_notional / float(self.eta)

        for s in winners:
            price = float(self.Securities[s].Price)
            if price <= 0:
                continue
            qty = long_notional_per / price
            if qty > 0:
                self.MarketOrder(s, qty)
                self.currentLongs.add(s)

        for s in losers:
            price = float(self.Securities[s].Price)
            if price <= 0:
                continue
            qty = short_notional_per / price
            if qty > 0:
                self.MarketOrder(s, -qty)
                self.currentShorts.add(s)

        # 4) Daily return accounting (Eq. 3): subtract fixed daily C once per day
        self._AccountDailyReturnWithCost()

        # 5) Track annualized return metric (Eq. 4) in a simple running way
        self._UpdateAnnualizedReturnMetric()

    def _UpdateDailyCloseHistory(self, data: Slice):
        # Daily resolution: bar.EndTime and self.Time align to the daily bar timestamp.
        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                continue
            bar = data.Bars[s]
            d = bar.EndTime.date()
            close = float(bar.Close)

            # Avoid duplicates for the same date
            if len(self.dailyDateHistory[s]) > 0 and self.dailyDateHistory[s][-1] == d:
                self.dailyCloseHistory[s][-1] = close
            else:
                self.dailyDateHistory[s].append(d)
                self.dailyCloseHistory[s].append(close)

    def _TryComputeTrainTestSplit(self):
        # Use common length across symbols; require aligned counts
        if len(self.symbols) == 0:
            return

        min_len = None
        for s in self.symbols:
            l = len(self.dailyCloseHistory[s])
            if l == 0:
                return
            min_len = l if min_len is None else min(min_len, l)

        # Need enough history for train + tau at least
        if min_len is None or min_len < max(30, self.tau + 5):
            return

        self.trainEndIndex = int(self.trainingFraction * min_len) - 1
        if self.trainEndIndex < self.tau + 1:
            self.trainEndIndex = None
            return

        # Define train end date using the common timeline from the first symbol
        # (we only use indices up to min_len, so each symbol has that date index)
        ref = self.symbols[0]
        self.trainEndDate = self.dailyDateHistory[ref][self.trainEndIndex]

    def _TryRunCausalDiscoveryOnTraining(self):
        # Implements a TRAIN-ONLY sparse parent selection to replace fully-connected placeholder.
        # This is structured as a "causal discovery phase" producing PA_X for each asset.
        # For self-cause-only control, PA_X = {X}.
        if self.trainEndIndex is None:
            return

        if self.useSelfCauseOnly:
            for x in self.symbols:
                self.parents[x] = [x]
            self.parentsComputed = True
            return

        # Data-driven lagged correlation "discovery" (no contemporaneous effect; use lag 1..tau)
        # Score each potential parent y->x by max abs corr between x_t and y_{t-lag} over lag 1..tau.
        # Then keep top maxParents (excluding x itself).
        n = self.trainEndIndex + 1
        if n <= self.tau + 5:
            return

        # Prepare training arrays truncated to common length n
        closes = {}
        for s in self.symbols:
            series = self.dailyCloseHistory[s]
            if len(series) < n:
                return
            closes[s] = np.asarray(series[:n], dtype=float)

        for x in self.symbols:
            x_series = closes[x]
            candidates = []
            for y in self.symbols:
                if y == x:
                    continue
                y_series = closes[y]

                best = 0.0
                for lag in range(1, self.tau + 1):
                    # correlate x[lag:] with y[:-lag]
                    a = x_series[lag:]
                    b = y_series[:-lag]
                    if len(a) < 5:
                        continue
                    a_std = float(np.std(a))
                    b_std = float(np.std(b))
                    if a_std <= 0 or b_std <= 0:
                        continue
                    c = float(np.corrcoef(a, b)[0, 1])
                    if not np.isfinite(c):
                        continue
                    if abs(c) > abs(best):
                        best = c

                candidates.append((y, abs(best)))

            candidates.sort(key=lambda kv: kv[1], reverse=True)
            chosen = [kv[0] for kv in candidates[:self.maxParents] if kv[1] > 0]
            if len(chosen) == 0:
                chosen = [x]  # fallback to self parent if no signal
            self.parents[x] = chosen

        # "Graph compression": our PA_X is already lag-attribute-free (symbols only),
        # but prediction will still use lags 1..tau in the regression features.
        self.parentsComputed = True

    def _ComputePredictedReturnsExpandingWindow(self):
        # Expanding window forecast within test, starting from training end.
        # Align lags consistently:
        # Training rows use lags 1..tau; prediction uses lags 1..tau as well (no lag 0).
        predicted = {}

        # Determine common min length and current index t
        min_len = None
        for s in self.symbols:
            l = len(self.dailyCloseHistory[s])
            if l == 0:
                return predicted
            min_len = l if min_len is None else min(min_len, l)

        if min_len is None:
            return predicted

        t = min_len - 1  # today's index
        if t <= self.trainEndIndex:
            return predicted

        # Use expanding training window from 0..t (as in paper, expanding during test),
        # but strictly causal (only past lags) and started after discovery was done on 0..trainEndIndex.
        for x in self.symbols:
            parents = self.parents.get(x, [])
            if parents is None or len(parents) == 0:
                continue

            px_series = self.dailyCloseHistory[x]
            if len(px_series) < min_len:
                continue

            feature_count = len(parents) * self.tau

            # Build rows s = tau..t with y_s = P_s^X, x_s = parents at s-1..s-tau
            # For stable OLS: require samples > k+1
            rows = []
            yvals = []
            for s_idx in range(self.tau, t + 1):
                feat = []
                ok = True
                for y in parents:
                    y_series = self.dailyCloseHistory[y]
                    if len(y_series) < min_len:
                        ok = False
                        break
                    for lag in range(1, self.tau + 1):
                        val = float(y_series[s_idx - lag])
                        feat.append(val)
                if not ok:
                    continue
                rows.append(feat)
                yvals.append(float(px_series[s_idx]))

            if len(rows) <= feature_count + 1:
                continue

            Xmat = np.asarray(rows, dtype=float)
            yvec = np.asarray(yvals, dtype=float)

            ones = np.ones((Xmat.shape[0], 1), dtype=float)
            A = np.concatenate([ones, Xmat], axis=1)

            try:
                beta, _, _, _ = np.linalg.lstsq(A, yvec, rcond=None)
            except:
                continue

            # Predict next day's price rho_{t+1} using features built from time t (known at end of day t),
            # with lags 1..tau mapping to P_t, P_{t-1}, ..., P_{t-tau+1}:
            feat_next = []
            for y in parents:
                y_series = self.dailyCloseHistory[y]
                for lag in range(1, self.tau + 1):
                    idx = t + 1 - lag
                    if idx < 0 or idx >= len(y_series):
                        feat_next = None
                        break
                    feat_next.append(float(y_series[idx]))
                if feat_next is None:
                    break
            if feat_next is None:
                continue

            x_next = np.asarray(feat_next, dtype=float)
            rho_next = float(beta[0] + np.dot(beta[1:], x_next))

            p_t = float(px_series[t])
            if p_t <= 0:
                continue

            gamma = (rho_next - p_t) / p_t
            predicted[x] = gamma

        return predicted

    def _CloseOutAllPositionsMarket(self):
        # Paper: close out at end of each day before next action.
        syms_to_check = set(self.currentLongs) | set(self.currentShorts)
        for s in self.symbols:
            if self.Portfolio[s].Invested:
                syms_to_check.add(s)

        for s in syms_to_check:
            p = self.Portfolio[s]
            if not p.Invested:
                continue
            qty_to_flatten = -float(p.Quantity)
            if qty_to_flatten != 0:
                self.MarketOrder(s, qty_to_flatten)

        self.currentLongs.clear()
        self.currentShorts.clear()

    def _AccountDailyReturnWithCost(self):
        # Realized daily portfolio return proxy:
        # use equity curve change from yesterday close to today close, then subtract fixed C once.
        # This matches Eq.(3) " - C " as a per-day deduction (accounting), regardless of turnover.
        equity = float(self.Portfolio.TotalPortfolioValue)
        if self.lastEquityValue is None:
            self.lastEquityValue = equity
            return

        if self.lastEquityValue <= 0:
            self.lastEquityValue = equity
            return

        raw_rp = (equity - self.lastEquityValue) / self.lastEquityValue
        rp_after_cost = raw_rp - float(self.transactionCostRate)

        self.dailyRp.append(float(rp_after_cost))
        self.lastEquityValue = equity

    def _UpdateAnnualizedReturnMetric(self):
        # Eq.(4): r_annual = (1 + r_Ttest)^(D/Ttest) - 1
        # Here we compute using accumulated dailyRp over the test days we have recorded.
        if len(self.dailyRp) < 2:
            return

        cumulative = 1.0
        for r in self.dailyRp:
            cumulative *= (1.0 + float(r))

        r_Ttest = cumulative - 1.0
        Ttest = float(len(self.dailyRp))
        D = 252.0

        annual = (1.0 + r_Ttest) ** (D / Ttest) - 1.0
        self.Plot("Metrics", "AnnualizedReturnEq4", float(annual))