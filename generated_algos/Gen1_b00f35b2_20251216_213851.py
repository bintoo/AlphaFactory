from AlgorithmImports import *
from datetime import timedelta, datetime

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.resolution = Resolution.Daily

        # Strategy parameters (Jegadeesh & Titman style)
        self.J_months = 12
        self.K_months = 3
        self.skip_week = True
        self.deciles = 10

        # Need enough history for J-month formation + 1-week skip + buffer
        # (daily data; 800 trading days is a safe upper bound for stable universe formation)
        self.SetWarmUp(timedelta(days=1200))

        # Universe + state
        self.symbols = []
        self.portfolio_queue = []
        self.next_monthly_formation_time = None

        # Universe selection (broad stock universe; monthly-ish via daily checks)
        self.universe_size = 500
        self.min_price = 5.0

        self.AddUniverse(self.CoarseSelectionFunction)

        self.RecoverState()

        if self.next_monthly_formation_time is None:
            self.next_monthly_formation_time = self._GetMonthStart(self.Time)

    def RecoverState(self):
        # Local safe mode: no persistence
        self.symbols = []
        self.portfolio_queue = []
        self.next_monthly_formation_time = None

    def CoarseSelectionFunction(self, coarse):
        # Build an investable, liquid universe of individual stocks (not ETFs)
        filtered = []
        for c in coarse:
            if not c.HasFundamentalData:
                continue
            if c.Price is None or c.Price <= self.min_price:
                continue
            if c.DollarVolume is None or c.DollarVolume <= 0:
                continue
            filtered.append(c)

        filtered.sort(key=lambda x: x.DollarVolume, reverse=True)
        selected = filtered[:self.universe_size]

        # Track selected symbols (no strings in algorithm logic elsewhere)
        self.symbols = [c.Symbol for c in selected]
        return self.symbols

    def OnData(self, data: Slice):
        # Scheduling must always run (exit logic is handled via combined targets every bar)
        if self.next_monthly_formation_time is None:
            self.next_monthly_formation_time = self._GetMonthStart(self.Time)

        should_form = self.Time >= self.next_monthly_formation_time

        # Formation/rebalance is evaluated monthly; orders are placed only when not warming up
        if should_form:
            formation_time = self.next_monthly_formation_time

            # Skip-week: ranking window must end at least ~1 week (5 trading days) before formation
            ranking_end_time = formation_time
            if self.skip_week:
                ranking_end_time = ranking_end_time - timedelta(days=7)

            ranking_end_month_start = self._GetMonthStart(ranking_end_time)
            start_time = self._AddMonths(ranking_end_month_start, -self.J_months)
            end_time = ranking_end_time

            eligible, returns_by_symbol = self._ComputeLaggedReturns(start_time, end_time)

            if len(eligible) >= self.deciles:
                ranked = sorted(eligible, key=lambda s: returns_by_symbol[s])
                groups = self._SplitIntoDeciles(ranked, self.deciles)

                if len(groups) == self.deciles and len(groups[0]) > 0 and len(groups[-1]) > 0:
                    losers = groups[0]
                    winners = groups[-1]

                    sp = {
                        'formation_time': formation_time,
                        'winner_symbols': winners,
                        'loser_symbols': losers,
                        'w_long': 0.0 if len(winners) == 0 else (1.0 / float(len(winners))),
                        'w_short': 0.0 if len(losers) == 0 else (-1.0 / float(len(losers)))
                    }
                    self.portfolio_queue.append(sp)

                    # Close out slice initiated in month t-K
                    if len(self.portfolio_queue) > self.K_months:
                        self.portfolio_queue.pop(0)

            self._AdvanceNextMonthlyFormation()

        # Aggregate overlapping sub-portfolios into one combined target book
        combined = {}
        for slice_sp in self.portfolio_queue:
            wL = slice_sp['w_long']
            wS = slice_sp['w_short']
            for s in slice_sp['winner_symbols']:
                combined[s] = combined.get(s, 0.0) + wL
            for s in slice_sp['loser_symbols']:
                combined[s] = combined.get(s, 0.0) + wS

        # Normalize by active vintages (startup months can have < K)
        active = len(self.portfolio_queue)
        if active > 0:
            for s in list(combined.keys()):
                combined[s] = combined[s] / float(active)

        # Ensure we exit any invested symbols not in current combined targets
        invested_symbols = []
        for kvp in self.Portfolio:
            sp = kvp.Value
            if sp is not None and sp.Invested:
                invested_symbols.append(kvp.Key)

        target_symbols = set(combined.keys())
        for s in invested_symbols:
            if s not in target_symbols:
                combined[s] = 0.0

        if not self.IsWarmingUp:
            self._RebalanceWithLimitOrders(data, combined)

    def _ComputeLaggedReturns(self, start_time, end_time):
        eligible = []
        returns_by_symbol = {}

        if self.symbols is None or len(self.symbols) == 0:
            return eligible, returns_by_symbol

        history = self.History(self.symbols, start_time, end_time, self.resolution)
        if history.empty:
            return eligible, returns_by_symbol

        for symbol in self.symbols:
            try:
                sym_hist = history.loc[symbol]
            except Exception:
                continue

            if sym_hist is None or sym_hist.empty:
                continue
            if 'close' not in sym_hist.columns:
                continue

            start_price = self._LastCloseOnOrBefore(sym_hist, start_time)
            end_price = self._LastCloseOnOrBefore(sym_hist, end_time)

            if start_price is None or end_price is None:
                continue
            if start_price <= 0:
                continue

            r = (end_price / start_price) - 1.0
            eligible.append(symbol)
            returns_by_symbol[symbol] = r

        return eligible, returns_by_symbol

    def _RebalanceWithLimitOrders(self, data: Slice, target_weights: dict):
        # Keep the original (paper-divergent) limit-order execution style, but avoid
        # excessive duplicates by only placing when we have a bar.
        portfolio_value = self.Portfolio.TotalPortfolioValue

        for symbol, target_w in target_weights.items():
            if not data.Bars.ContainsKey(symbol):
                continue

            price = data.Bars[symbol].Close
            if price is None or price <= 0:
                continue

            target_value = portfolio_value * target_w
            target_qty = target_value / price

            current_qty = self.Portfolio[symbol].Quantity
            delta = target_qty - current_qty

            if abs(delta) < 1:
                continue

            self.LimitOrder(symbol, int(delta), price)

    def _SplitIntoDeciles(self, ranked_symbols, deciles: int):
        n = len(ranked_symbols)
        groups = []
        base = n // deciles
        rem = n % deciles

        idx = 0
        for d in range(deciles):
            size = base + (1 if d < rem else 0)
            groups.append(ranked_symbols[idx: idx + size])
            idx += size
        return groups

    def _LastCloseOnOrBefore(self, sym_hist, t):
        try:
            filtered = sym_hist.loc[:t]
        except Exception:
            return None

        if filtered is None or filtered.empty:
            return None
        if 'close' not in filtered.columns:
            return None

        val = filtered.iloc[-1]['close']
        if val is None:
            return None
        return float(val)

    def _GetMonthStart(self, t):
        return datetime(t.year, t.month, 1)

    def _AddMonths(self, t, months):
        year = t.year
        month = t.month + months
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        return datetime(year, month, 1)

    def _AdvanceNextMonthlyFormation(self):
        if self.next_monthly_formation_time is None:
            self.next_monthly_formation_time = self._GetMonthStart(self.Time)
            return
        self.next_monthly_formation_time = self._AddMonths(self.next_monthly_formation_time, 1)