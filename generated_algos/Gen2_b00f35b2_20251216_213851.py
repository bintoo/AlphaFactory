from AlgorithmImports import *
from datetime import timedelta, datetime

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.resolution = Resolution.Daily
        self.SetWarmUp(timedelta(days=35))

        # Strategy parameters
        self.J_months = 12
        self.K_months = 3
        self.skip_week = True
        self.deciles = 10

        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        self.symbols = []
        for ticker in tickers:
            equity = self.AddEquity(ticker, self.resolution)
            self.symbols.append(equity.Symbol)

        self.portfolio_queue = []
        self.next_monthly_formation_time = None

        self.RecoverState()

        if self.next_monthly_formation_time is None:
            self.next_monthly_formation_time = self._GetMonthStart(self.Time)

    def RecoverState(self):
        # Local safe mode: no persistence
        self.portfolio_queue = []
        self.next_monthly_formation_time = None

    def OnData(self, data: Slice):
        # Always run exit logic checks / scheduling checks each bar (no early return gating)
        if self.next_monthly_formation_time is None:
            self.next_monthly_formation_time = self._GetMonthStart(self.Time)

        # Rebalance/formation is monthly, but OnData must not return early and skip exit logic.
        should_form = self.Time >= self.next_monthly_formation_time

        if should_form:
            formation_time = self.next_monthly_formation_time

            end_time = formation_time
            if self.skip_week:
                end_time = end_time - timedelta(days=7)

            # Month-aligned start for the J-month lookback window
            start_time = self._AddMonths(self._GetMonthStart(end_time), -self.J_months)

            history = self.History(self.symbols, start_time, end_time, self.resolution)

            eligible = []
            returns_by_symbol = {}

            if not history.empty:
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

                    if len(self.portfolio_queue) > self.K_months:
                        self.portfolio_queue.pop(0)

            # Whether formation succeeded or not, advance schedule so we don't repeatedly try on every bar
            self._AdvanceNextMonthlyFormation()

        # Exit/rebalance targets should be computed and (if not warming up) enforced every bar
        # so exits are not gated to the monthly formation branch.
        combined = {}
        for slice_sp in self.portfolio_queue:
            wL = slice_sp['w_long']
            wS = slice_sp['w_short']
            for s in slice_sp['winner_symbols']:
                combined[s] = combined.get(s, 0.0) + wL
            for s in slice_sp['loser_symbols']:
                combined[s] = combined.get(s, 0.0) + wS

        # Deterministic normalization (stable exposure across overlapping K slices)
        if self.K_months > 0:
            for s in list(combined.keys()):
                combined[s] = combined[s] / float(self.K_months)

        target_symbols = set(combined.keys())
        for s in self.symbols:
            if s not in target_symbols and self.Portfolio[s].Invested:
                combined[s] = 0.0

        if not self.IsWarmingUp:
            self._RebalanceWithLimitOrders(data, combined)

    def _RebalanceWithLimitOrders(self, data: Slice, target_weights: dict):
        portfolio_value = self.Portfolio.TotalPortfolioValue

        for symbol, target_w in target_weights.items():
            if not data.Bars.ContainsKey(symbol):
                continue

            price = data.Bars[symbol].Close
            if price <= 0:
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