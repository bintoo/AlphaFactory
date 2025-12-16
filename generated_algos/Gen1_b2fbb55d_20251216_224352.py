from AlgorithmImports import *
import numpy as np
from collections import deque

class MyAgentStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Need up to 60 months of trailing monthly returns; use ~6 years warmup for safety
        self.SetWarmUp(timedelta(days=6 * 365))

        # Universe (fixed tickers per instruction)
        self.symbols = []
        for ticker in ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # Market proxy for beta (use SPY if present, else first symbol)
        self.market = None
        for s in self.symbols:
            if s.Value == "SPY":
                self.market = s
                break
        if self.market is None:
            self.market = self.symbols[0]

        # Strategy configuration (choose one)
        self.Mode = "MinVarianceDiagonal"
        # Allowed: "VolatilityQuintileLow", "BetaQuintileLow", "MinVarianceDiagonal", "LeveredMinVarianceBeta1"

        # Rolling monthly return storage (up to 60 months, need at least 24)
        self.MinHistoryMonths = 24
        self.LookbackMonths = 60

        # Persistent state containers (must be recovered)
        self.returns = {}                # Symbol -> deque of monthly returns
        self.marketReturns = deque(maxlen=self.LookbackMonths)

        self.lastMonthClose = {}         # Symbol -> last observed EOM close used for monthly return calc
        self.lastMarketClose = None

        self.currentTargets = {}         # Symbol -> target weight (desired)
        self.lastRebalanceMonthKey = None

        self.RecoverState()

    def RecoverState(self):
        if self.returns is None:
            self.returns = {}
        if self.marketReturns is None:
            self.marketReturns = deque(maxlen=self.LookbackMonths)
        if self.lastMonthClose is None:
            self.lastMonthClose = {}
        if self.currentTargets is None:
            self.currentTargets = {}

        total_value = float(self.Portfolio.TotalPortfolioValue)
        if total_value <= 0:
            return

        recovered_targets = {}
        for s in self.symbols:
            p = self.Portfolio[s]
            if not p.Invested:
                continue
            holding_value = float(p.HoldingsValue)
            if abs(holding_value) <= 0:
                continue
            recovered_targets[s] = holding_value / total_value

        self.currentTargets = recovered_targets

    def OnData(self, data: Slice):
        # Always run exit logic (even during warmup)
        self.ManageExits(data)

        # Need market bar for date context and EOM logic
        if not data.Bars.ContainsKey(self.market):
            return

        time = self.Time
        month_key = time.year * 100 + time.month

        # Only act at month-end (last trading day of month)
        if not self.IsEndOfMonth(time):
            return

        # Prevent multiple runs within the same month-end day
        if self.lastRebalanceMonthKey == month_key:
            return

        # 1) Update monthly returns (for all symbols) using EOM closes
        self.UpdateMonthlyReturns(data)

        # 2) Attempt rebalance (disable ordering during warmup)
        if not self.IsWarmingUp:
            targets = self.ComputeTargets(data)
            if targets is not None:
                self.RebalanceToTargets(data, targets)
                self.currentTargets = targets

        self.lastRebalanceMonthKey = month_key

    def ManageExits(self, data: Slice):
        # Paper logic does not specify stop-loss/exit rules; exits occur via monthly rebalance.
        return

    def IsEndOfMonth(self, time: datetime) -> bool:
        # For daily data: if the next trading day is in a different month, it's month-end.
        # Use the exchange hours helper.
        next_open_dt = self.Securities[self.market].Exchange.Hours.GetNextMarketOpen(time, False)
        next_trading_day = next_open_dt.date()
        return next_trading_day.month != time.month

    def UpdateMonthlyReturns(self, data: Slice):
        # Update market return first
        if data.Bars.ContainsKey(self.market):
            m_close = float(data.Bars[self.market].Close)
            if self.lastMarketClose is not None and self.lastMarketClose > 0:
                r_m = (m_close / self.lastMarketClose) - 1.0
                self.marketReturns.append(float(r_m))
            self.lastMarketClose = m_close

        # Update each symbol return
        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                continue
            close = float(data.Bars[s].Close)
            prev = self.lastMonthClose.get(s, None)
            if prev is not None and prev > 0:
                r = (close / prev) - 1.0
                if s not in self.returns or self.returns[s] is None:
                    self.returns[s] = deque(maxlen=self.LookbackMonths)
                self.returns[s].append(float(r))
            self.lastMonthClose[s] = close

    def _GetQuintileBucketIndices(self, n: int, bucket_count: int = 5):
        # Returns list of (start, end) index pairs, covering [0, n)
        # Buckets differ by at most 1, and all items are assigned.
        if n <= 0:
            return []
        base = n // bucket_count
        rem = n % bucket_count
        buckets = []
        start = 0
        for i in range(bucket_count):
            size = base + (1 if i < rem else 0)
            end = start + size
            buckets.append((start, end))
            start = end
        return buckets

    def ComputeTargets(self, data: Slice):
        if len(self.marketReturns) < self.MinHistoryMonths:
            return None

        eligible = []
        for s in self.symbols:
            if s not in self.returns:
                continue
            if len(self.returns[s]) < self.MinHistoryMonths:
                continue
            if not data.Bars.ContainsKey(s):
                continue
            eligible.append(s)

        if len(eligible) == 0:
            return None

        mode = self.Mode

        market_arr_full = np.array(list(self.marketReturns), dtype=float)
        if market_arr_full.size < self.MinHistoryMonths:
            return None

        vols = {}
        betas = {}

        for s in eligible:
            r_full = np.array(list(self.returns[s]), dtype=float)
            if r_full.size < self.MinHistoryMonths:
                continue

            # Use up to 60 months (already capped by deque), but keep explicit L and align with market
            L = min(int(r_full.size), int(market_arr_full.size), int(self.LookbackMonths))
            if L < self.MinHistoryMonths:
                continue

            r = r_full[-L:]
            rm = market_arr_full[-L:]

            # Volatility (sample std dev)
            if r.size >= 2:
                vols[s] = float(np.std(r, ddof=1))
            else:
                continue

            # Beta = Cov(ri, rm) / Var(rm)
            if rm.size >= 2:
                var_rm = float(np.var(rm, ddof=1))
                if var_rm > 0:
                    cov = float(np.cov(r, rm, ddof=1)[0, 1])
                    betas[s] = cov / var_rm
                else:
                    betas[s] = 0.0
            else:
                betas[s] = 0.0

        eligible = [s for s in eligible if s in vols and s in betas]
        if len(eligible) < 2:
            return None

        if mode == "VolatilityQuintileLow":
            sorted_syms = sorted(eligible, key=lambda x: vols.get(x, float('inf')))
            buckets = self._GetQuintileBucketIndices(len(sorted_syms), 5)
            if len(buckets) < 1:
                return None
            start, end = buckets[0]
            qlow = sorted_syms[start:end]
            if len(qlow) == 0:
                return None
            w = 1.0 / float(len(qlow))
            return {s: w for s in qlow}

        if mode == "BetaQuintileLow":
            sorted_syms = sorted(eligible, key=lambda x: betas.get(x, float('inf')))
            buckets = self._GetQuintileBucketIndices(len(sorted_syms), 5)
            if len(buckets) < 1:
                return None
            start, end = buckets[0]
            qlow = sorted_syms[start:end]
            if len(qlow) == 0:
                return None
            w = 1.0 / float(len(qlow))
            return {s: w for s in qlow}

        if mode in ["MinVarianceDiagonal", "LeveredMinVarianceBeta1"]:
            sigma2 = {}
            for s in eligible:
                v = vols.get(s, None)
                if v is None or v <= 0:
                    continue
                sigma2[s] = v * v

            if len(sigma2) == 0:
                return None

            max_w = 0.03
            free = set(sigma2.keys())
            fixed = {}
            remaining = 1.0

            # Iterative capping + renormalization
            while True:
                if remaining <= 1e-12:
                    # No weight left; zero out remaining free set
                    for s in list(free):
                        fixed[s] = 0.0
                    free.clear()
                    break

                if len(free) == 0:
                    break

                inv = {s: 1.0 / sigma2[s] for s in free}
                denom = sum(inv.values())
                if denom <= 0:
                    # Cannot allocate remaining; set to 0
                    for s in list(free):
                        fixed[s] = 0.0
                    free.clear()
                    break

                tentative = {s: remaining * (inv[s] / denom) for s in free}

                violators = [s for s, tw in tentative.items() if tw > max_w]
                if len(violators) == 0:
                    for s, tw in tentative.items():
                        fixed[s] = float(tw)
                    free.clear()
                    remaining = 0.0
                    break

                # Cap violators at max_w
                for s in violators:
                    fixed[s] = float(max_w)
                    if s in free:
                        free.remove(s)
                    remaining -= max_w

            weights = fixed

            # Optional leverage to beta 1
            if mode == "LeveredMinVarianceBeta1":
                beta_p = 0.0
                for s, w in weights.items():
                    beta_p += float(w) * float(betas.get(s, 0.0))

                if abs(beta_p) > 1e-8:
                    lev = 1.0 / beta_p
                    weights = {s: float(w) * float(lev) for s, w in weights.items()}

            # Drop near-zero weights
            weights = {s: float(w) for s, w in weights.items() if abs(float(w)) > 1e-8}
            if len(weights) == 0:
                return None
            return weights

        return None

    def RebalanceToTargets(self, data: Slice, targets: dict):
        # Paper-style monthly rebalance; execution via LimitOrder (no MarketOrders).

        portfolio_value = float(self.Portfolio.TotalPortfolioValue)
        if portfolio_value <= 0:
            return

        # Desired quantities for target holdings
        desired_qty = {}
        for s, w in targets.items():
            if not data.Bars.ContainsKey(s):
                continue
            price = float(data.Bars[s].Close)
            if price <= 0:
                continue
            target_value = portfolio_value * float(w)
            qty = int(target_value / price)
            desired_qty[s] = qty

        # Liquidate symbols not in targets (sell current quantity with a limit order)
        for s in self.symbols:
            p = self.Portfolio[s]
            if not p.Invested:
                continue
            if s in targets:
                continue
            if not data.Bars.ContainsKey(s):
                continue
            price = float(data.Bars[s].Close)
            current_qty = int(p.Quantity)
            if current_qty != 0:
                self.LimitOrder(s, -current_qty, float(price))

        # Adjust holdings for target symbols
        for s, tgt_qty in desired_qty.items():
            if not data.Bars.ContainsKey(s):
                continue
            price = float(data.Bars[s].Close)
            cur_qty = int(self.Portfolio[s].Quantity)
            delta = int(tgt_qty - cur_qty)
            if delta == 0:
                continue

            # Conservative limit pricing:
            if delta > 0:
                limit_price = price * 1.001
            else:
                limit_price = price * 0.999

            self.LimitOrder(s, delta, float(limit_price))