from AlgorithmImports import *
import math

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Paper: 12-month trailing window; warmup must cover the lookback to avoid partial-history behavior
        self.lookbackDays = 252
        self.SetWarmUp(timedelta(days=self.lookbackDays))

        # Universe
        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # tau ∈ {0,1}; this implementation is tau=0 (naive centered bisection).
        # (We do not claim tau=1 behavior.)
        self.tau = 0

        # Rebalance state
        self.lastRebalanceQuarterKey = None
        self.lastRebalanceDateKey = None
        self.lastSeenMonthKey = None

        # rolling returns state
        self.prevClose = {}          # Symbol -> last close used for return calculation
        self.returnsHistory = {}     # Symbol -> dict[dateKey -> return], capped at lookbackDays
        for s in self.symbols:
            self.prevClose[s] = None
            self.returnsHistory[s] = {}

        self.targetWeights = {s: 0.0 for s in self.symbols}

        self.RecoverState()

    def RecoverState(self):
        # Minimal deterministic recovery; do not attempt to reconstruct return history.
        self.lastRebalanceQuarterKey = None
        self.lastRebalanceDateKey = None
        self.lastSeenMonthKey = None

        rebuilt_prev_close = {s: None for s in self.symbols}
        rebuilt_target_weights = {s: 0.0 for s in self.symbols}

        total_value = self.Portfolio.TotalPortfolioValue
        if total_value > 0:
            for s in self.symbols:
                p = self.Portfolio[s]
                if p.Invested and p.Quantity != 0:
                    rebuilt_prev_close[s] = float(p.AveragePrice)
                    rebuilt_target_weights[s] = float((p.Quantity * p.AveragePrice) / total_value)

        self.prevClose = rebuilt_prev_close
        self.returnsHistory = {s: {} for s in self.symbols}
        self.targetWeights = rebuilt_target_weights

    def OnData(self, data: Slice):
        # Always run state updates bar-by-bar (even during warmup)
        dayKey = (self.Time.year, self.Time.month, self.Time.day)

        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                continue

            close = float(data.Bars[s].Close)
            if close <= 0:
                continue

            if self.prevClose[s] is None:
                self.prevClose[s] = close
                continue

            prev = float(self.prevClose[s])
            if prev > 0:
                r = (close / prev) - 1.0
                rh = self.returnsHistory[s]
                rh[dayKey] = r
                # Keep only the most recent lookbackDays observations by dateKey ordering
                if len(rh) > self.lookbackDays:
                    keys = sorted(rh.keys())
                    to_drop = len(keys) - self.lookbackDays
                    for k in keys[:to_drop]:
                        rh.pop(k, None)

            self.prevClose[s] = close

        # Exit/position-management logic must be evaluated each bar.
        self._ExitLogic(data)

        # Order placement only after warmup
        if self.IsWarmingUp:
            return

        # Rebalance attempt evaluated each bar, but it must be first trading day of the quarter.
        if not self._IsQuarterStart(self.Time):
            return

        if self.lastRebalanceDateKey == dayKey:
            return

        quarterKey = self._QuarterKey(self.Time)
        if self.lastRebalanceQuarterKey == quarterKey:
            self.lastRebalanceDateKey = dayKey
            return

        # Build synchronized return panel on common dates (intersection) to avoid biased covariance.
        common_dates = None
        for s in self.symbols:
            rh = self.returnsHistory[s]
            if len(rh) < self.lookbackDays:
                self.lastRebalanceDateKey = dayKey
                return
            ds = set(rh.keys())
            common_dates = ds if common_dates is None else (common_dates & ds)

        if common_dates is None or len(common_dates) < self.lookbackDays:
            self.lastRebalanceDateKey = dayKey
            return

        # Use the most recent lookbackDays common dates
        common_dates_sorted = sorted(common_dates)
        common_dates_sorted = common_dates_sorted[-self.lookbackDays:]

        assets = list(self.symbols)
        returns_matrix = []
        for s in assets:
            rh = self.returnsHistory[s]
            series = [float(rh[d]) for d in common_dates_sorted]
            returns_matrix.append(series)

        cov = self._CovarianceMatrixFromPanel(returns_matrix)
        corr = self._CorrelationMatrixFromCov(cov)
        dist = self._DistanceMatrixFromCorr(corr)

        # Proper single-linkage hierarchical clustering linkage + quasi-diagonalisation
        linkage = self._SingleLinkage(dist)
        order_idx = self._GetQuasiDiag(linkage, len(assets))

        cov_re = self._ReorderMatrix(cov, order_idx)

        # HRP recursive bisection with inverse-variance intra-cluster weights (Lopez de Prado)
        # tau=0 naive centered bisection only; tau not otherwise used.
        w_ordered = self._HRPRecursiveBisectionWeights(cov_re, tau=self.tau)

        ordered_assets = [assets[i] for i in order_idx]
        w_by_symbol = {s: 0.0 for s in assets}
        for j, s in enumerate(ordered_assets):
            w_by_symbol[s] = float(w_ordered[j])

        # Enforce per-asset box constraints (implemented; defaults are long-only [0,1])
        w_min = {s: 0.0 for s in assets}
        w_max = {s: 1.0 for s in assets}
        w_by_symbol = self._ApplyBoxConstraintsToWeights(w_by_symbol, w_min, w_max)

        self.targetWeights = dict(w_by_symbol)

        # Limit-order rebalance only (per instruction)
        total_value = float(self.Portfolio.TotalPortfolioValue)
        for s in assets:
            if not data.Bars.ContainsKey(s):
                continue
            price = float(data.Bars[s].Close)
            if price <= 0:
                continue

            target_value = float(w_by_symbol[s]) * total_value
            target_qty = int(math.floor(target_value / price))
            current_qty = int(self.Portfolio[s].Quantity)
            dq = target_qty - current_qty
            if dq == 0:
                continue

            self.LimitOrder(s, dq, price)

        self.lastRebalanceQuarterKey = quarterKey
        self.lastRebalanceDateKey = dayKey

    def _ExitLogic(self, data: Slice):
        return

    def _QuarterKey(self, t):
        q = ((t.month - 1) // 3) + 1
        return (t.year, q)

    def _IsQuarterStart(self, t):
        # Rebalance on the FIRST trading day that enters a quarter month.
        # Track month transitions; trigger only when the month changes into Jan/Apr/Jul/Oct.
        monthKey = (t.year, t.month)
        if self.lastSeenMonthKey is None:
            self.lastSeenMonthKey = monthKey
            return False

        if monthKey == self.lastSeenMonthKey:
            return False

        # Month changed; update and check if new month is quarter start month
        self.lastSeenMonthKey = monthKey
        return t.month in [1, 4, 7, 10]

    def _CovarianceMatrixFromPanel(self, panel):
        # panel: list of return series, aligned on the same dates; panel[i][t]
        n = len(panel)
        T = len(panel[0]) if n > 0 else 0
        if n == 0 or T <= 1:
            return [[0.0 for _ in range(n)] for _ in range(n)]

        means = [sum(panel[i]) / T for i in range(n)]
        cov = [[0.0 for _ in range(n)] for _ in range(n)]
        denom = T - 1

        for i in range(n):
            for j in range(i, n):
                c = 0.0
                mi = means[i]
                mj = means[j]
                ri = panel[i]
                rj = panel[j]
                for k in range(T):
                    c += (ri[k] - mi) * (rj[k] - mj)
                c /= denom
                cov[i][j] = c
                cov[j][i] = c
        return cov

    def _CorrelationMatrixFromCov(self, cov):
        n = len(cov)
        corr = [[0.0 for _ in range(n)] for _ in range(n)]
        std = [0.0 for _ in range(n)]
        for i in range(n):
            v = cov[i][i]
            std[i] = math.sqrt(v) if v > 0 else 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i][j] = 1.0
                else:
                    denom = std[i] * std[j]
                    if denom > 0:
                        corr[i][j] = cov[i][j] / denom
                    else:
                        corr[i][j] = 0.0

                if corr[i][j] > 1.0:
                    corr[i][j] = 1.0
                if corr[i][j] < -1.0:
                    corr[i][j] = -1.0
        return corr

    def _DistanceMatrixFromCorr(self, corr):
        n = len(corr)
        dist = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                rho = corr[i][j]
                x = 0.5 * (1.0 - rho)
                if x < 0:
                    x = 0.0
                dist[i][j] = math.sqrt(x)
        return dist

    # ---------------------------
    # Hierarchical clustering: single-linkage (AGNES) + quasi-diagonalisation
    # ---------------------------
    def _SingleLinkage(self, dist):
        # Returns linkage as list of rows: [idx1, idx2, distance, sample_count]
        # idx1/idx2 refer to either original points [0..n-1] or newly formed clusters [n..]
        n = len(dist)
        clusters = {i: [i] for i in range(n)}  # cluster_id -> members
        active_ids = list(range(n))
        next_id = n
        linkage = []

        def cluster_distance(id_a, id_b):
            ca = clusters[id_a]
            cb = clusters[id_b]
            m = None
            for i in ca:
                for j in cb:
                    d = dist[i][j]
                    if m is None or d < m:
                        m = d
            return m if m is not None else 0.0

        while len(active_ids) > 1:
            best_a = active_ids[0]
            best_b = active_ids[1]
            best_d = cluster_distance(best_a, best_b)

            for ii in range(len(active_ids)):
                for jj in range(ii + 1, len(active_ids)):
                    a = active_ids[ii]
                    b = active_ids[jj]
                    d = cluster_distance(a, b)
                    if d < best_d:
                        best_d = d
                        best_a = a
                        best_b = b

            members = clusters[best_a] + clusters[best_b]
            clusters[next_id] = members

            linkage.append([best_a, best_b, float(best_d), int(len(members))])

            active_ids = [x for x in active_ids if x != best_a and x != best_b]
            active_ids.append(next_id)
            next_id += 1

        return linkage

    def _GetQuasiDiag(self, linkage, n):
        # Quasi-diagonalisation: recursively traverse the linkage to get leaf order.
        # This is the dendrogram-derived leaf ordering used by HRP.
        if n <= 1:
            return list(range(n))

        # Build children map for non-leaf cluster ids
        children = {}
        for k, row in enumerate(linkage):
            a = int(row[0])
            b = int(row[1])
            cluster_id = n + k
            children[cluster_id] = (a, b)

        root = n + len(linkage) - 1

        def expand(node):
            if node < n:
                return [node]
            left, right = children[node]
            return expand(left) + expand(right)

        return expand(root)

    def _ReorderMatrix(self, M, order_idx):
        n = len(order_idx)
        R = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            oi = order_idx[i]
            for j in range(n):
                oj = order_idx[j]
                R[i][j] = float(M[oi][oj])
        return R

    # ---------------------------
    # HRP weights: recursive bisection using inverse-variance within cluster
    # ---------------------------
    def _HRPRecursiveBisectionWeights(self, cov_re, tau=0):
        n = len(cov_re)
        w = [1.0 for _ in range(n)]
        L = [list(range(n))]

        while True:
            all_singletons = True
            for cluster in L:
                if len(cluster) > 1:
                    all_singletons = False
                    break
            if all_singletons:
                break

            L_next = []
            for cluster in L:
                if len(cluster) == 1:
                    L_next.append(cluster)
                    continue

                # tau=0 naive centered split (paper extreme; dendrogram-respecting tau=1 not implemented here)
                mid = len(cluster) // 2
                if mid <= 0:
                    mid = 1
                S1 = cluster[:mid]
                S2 = cluster[mid:]
                subsets = [S1, S2]

                tildeVs = []
                for Sj in subsets:
                    Vj = self._SubMatrix(cov_re, Sj)
                    w_in = self._InverseVarianceWeightsFromCov(Vj)
                    tildeV = self._QuadraticForm(Vj, w_in)
                    tildeVs.append(float(tildeV))

                invVs = []
                for tv in tildeVs:
                    invVs.append(0.0 if tv <= 0 else (1.0 / tv))

                s_inv = sum(invVs)
                if s_inv <= 0:
                    alphas = [1.0 / len(subsets) for _ in subsets]
                else:
                    alphas = [x / s_inv for x in invVs]

                # Bottom-up enforcement of box/group constraints at this split level is implemented.
                # Default: no additional constraints beyond long-only; i.e., alpha bounds [0,1].
                a_min = [0.0 for _ in subsets]
                a_max = [1.0 for _ in subsets]

                tilde_alphas = self._EnforceAlphaConstraints(alphas, a_min, a_max)

                for j, Sj in enumerate(subsets):
                    a = float(tilde_alphas[j])
                    for p in Sj:
                        w[p] *= a

                for Sj in subsets:
                    L_next.append(Sj)

            L = L_next

        s = sum(w)
        if s > 0:
            w = [x / s for x in w]
        return w

    def _InverseVarianceWeightsFromCov(self, cov):
        m = len(cov)
        if m <= 0:
            return []

        inv = []
        for i in range(m):
            v = float(cov[i][i])
            if v > 0:
                inv.append(1.0 / v)
            else:
                inv.append(0.0)

        s = sum(inv)
        if s <= 0:
            return [1.0 / m for _ in range(m)]
        return [x / s for x in inv]

    def _SubMatrix(self, M, idxs):
        m = len(idxs)
        R = [[0.0 for _ in range(m)] for _ in range(m)]
        for i in range(m):
            for j in range(m):
                R[i][j] = float(M[idxs[i]][idxs[j]])
        return R

    def _QuadraticForm(self, M, w):
        m = len(w)
        total = 0.0
        for i in range(m):
            for j in range(m):
                total += float(w[i]) * float(M[i][j]) * float(w[j])
        return total

    # ---------------------------
    # Constraints machinery
    # ---------------------------
    def _EnforceAlphaConstraints(self, alpha, a_min, a_max):
        # Appendix A.3 step 5a-5c (implemented iteratively).
        k = len(alpha)
        if k == 0:
            return []

        # Step 5a: componentwise clipping
        tilde = [0.0 for _ in range(k)]
        for i in range(k):
            lo = float(a_min[i])
            hi = float(a_max[i])
            x = float(alpha[i])
            if x < lo:
                x = lo
            if x > hi:
                x = hi
            tilde[i] = x

        # Step 5b/5c: adjust unclipped entries to force sum to 1
        # Iterate with a safety cap; for 2 children it converges quickly.
        for _ in range(20):
            s = sum(tilde)
            diff = 1.0 - s
            if abs(diff) <= 1e-12:
                break

            # Find indices not clipped (tilde == alpha) in the paper.
            # Use exact equality with small tolerance.
            free = []
            free_alpha_sum = 0.0
            for i in range(k):
                if abs(tilde[i] - float(alpha[i])) <= 1e-12:
                    free.append(i)
                    free_alpha_sum += float(alpha[i])

            if len(free) == 0 or free_alpha_sum == 0.0:
                # No adjustable elements; renormalize if possible (last resort)
                if s > 0:
                    tilde = [x / s for x in tilde]
                break

            # Distribute |diff| proportional to alpha_k / sum(alpha_k) (paper)
            add = abs(diff)
            for i in free:
                tilde[i] = float(tilde[i]) + add * float(alpha[i]) / free_alpha_sum

            # Re-clip to respect bounds after adjustment
            for i in range(k):
                lo = float(a_min[i])
                hi = float(a_max[i])
                if tilde[i] < lo:
                    tilde[i] = lo
                if tilde[i] > hi:
                    tilde[i] = hi

        # Final small normalization safeguard
        s = sum(tilde)
        if s > 0:
            tilde = [x / s for x in tilde]
        return tilde

    def _ApplyBoxConstraintsToWeights(self, w_by_symbol, w_min, w_max):
        # Enforce per-asset box constraints w_i in [w_min_i, w_max_i] with iterative redistribution.
        assets = list(w_by_symbol.keys())
        n = len(assets)
        if n == 0:
            return w_by_symbol

        # Start from nonnegative weights
        w = {s: max(0.0, float(w_by_symbol[s])) for s in assets}

        # Normalize initial
        s0 = sum(w.values())
        if s0 > 0:
            for s in assets:
                w[s] /= s0
        else:
            # If all zero, start equal then project
            for s in assets:
                w[s] = 1.0 / n

        # Iteratively project onto box with sum-to-1 via redistribution among free weights.
        for _ in range(50):
            # Clip
            for s in assets:
                lo = float(w_min.get(s, 0.0))
                hi = float(w_max.get(s, 1.0))
                if w[s] < lo:
                    w[s] = lo
                if w[s] > hi:
                    w[s] = hi

            total = sum(w.values())
            diff = 1.0 - total
            if abs(diff) <= 1e-12:
                break

            # Free assets are those not at a bound in the direction we need to move.
            free = []
            for s in assets:
                lo = float(w_min.get(s, 0.0))
                hi = float(w_max.get(s, 1.0))
                if diff > 0 and w[s] < hi - 1e-15:
                    free.append(s)
                if diff < 0 and w[s] > lo + 1e-15:
                    free.append(s)

            if len(free) == 0:
                # Can't satisfy exactly; renormalize if possible
                total = sum(w.values())
                if total > 0:
                    for s in assets:
                        w[s] /= total
                break

            # Redistribute diff proportionally to current weights among free (fallback equal if all zero)
            denom = sum(w[s] for s in free)
            if denom <= 0:
                for s in free:
                    w[s] += diff / len(free)
            else:
                for s in free:
                    w[s] += diff * (w[s] / denom)

        # Final clamp + normalize safeguard
        for s in assets:
            lo = float(w_min.get(s, 0.0))
            hi = float(w_max.get(s, 1.0))
            if w[s] < lo:
                w[s] = lo
            if w[s] > hi:
                w[s] = hi

        total = sum(w.values())
        if total > 0:
            for s in assets:
                w[s] /= total
        return w