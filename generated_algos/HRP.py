from AlgorithmImports import *
import numpy as np
from collections import deque

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.SetWarmUp(timedelta(days=30))

        self.lookbackDays = 252
        self.lastRebalanceTime = datetime.min

        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        self.returns = {s: deque(maxlen=self.lookbackDays) for s in self.symbols}
        self.prevClose = {s: None for s in self.symbols}

        self.assetMin = {s: 0.0 for s in self.symbols}
        self.assetMax = {s: 1.0 for s in self.symbols}

        self.groupId = {s: str(s) for s in self.symbols}
        self.groupMin = {self.groupId[s]: 0.0 for s in self.symbols}
        self.groupMax = {self.groupId[s]: 1.0 for s in self.symbols}

        self.currentTargetWeights = {s: 0.0 for s in self.symbols}

        self.openOrderIdsBySymbol = {s: [] for s in self.symbols}

        self.RecoverState()

    def RecoverState(self):
        history = self.History(self.symbols, self.lookbackDays + 1, Resolution.Daily)
        if history is None or history.empty:
            return

        for s in self.symbols:
            try:
                df = history.loc[s]
            except Exception:
                continue

            if df is None or df.empty:
                continue
            if 'close' not in df.columns:
                continue

            closes = df['close'].dropna()
            if len(closes) < 2:
                continue

            self.returns[s].clear()
            prev = None
            for _, c in closes.items():
                if prev is None:
                    prev = float(c)
                    continue
                c = float(c)
                if prev != 0:
                    r = c / prev - 1.0
                    self.returns[s].append(float(r))
                prev = c

            self.prevClose[s] = float(closes.iloc[-1])

    def OnData(self, data: Slice):
        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                continue
            bar = data.Bars[s]
            close = float(bar.Close)
            prev = self.prevClose[s]
            if prev is not None and prev != 0:
                r = close / prev - 1.0
                self.returns[s].append(float(r))
            self.prevClose[s] = close

        self.ManageOpenOrders()

        if self.IsWarmingUp:
            return

        if not self.IsQuarterStart(self.Time):
            return
        if self.Time <= self.lastRebalanceTime:
            return
        if not self.AllReadyForCovariance():
            return

        assets = list(self.symbols)
        ret_matrix = self.BuildReturnMatrix(assets)
        if ret_matrix is None:
            return

        cov = np.cov(ret_matrix, rowvar=False, ddof=1)
        if cov is None or cov.shape[0] != len(assets):
            return

        corr = self.CovToCorr(cov)
        dist = np.sqrt(0.5 * (1.0 - corr))

        ordered_assets = self.SingleLinkageOrder(assets, dist)
        if ordered_assets is None or len(ordered_assets) != len(assets):
            return

        weights = self.HRPWeights(ordered_assets, cov, assets)
        if weights is None:
            return

        ssum = sum(weights.values())
        if ssum <= 0:
            return
        for s in weights:
            weights[s] = float(weights[s] / ssum)

        self.currentTargetWeights = dict(weights)

        self.RebalanceWithLimitOrders(weights)

        self.lastRebalanceTime = self.Time

    def CancelOrder(self, orderId: int):
        try:
            self.Transactions.CancelOrder(orderId)
        except Exception:
            pass

    def ManageOpenOrders(self):
        for s in self.symbols:
            ids = self.openOrderIdsBySymbol.get(s, [])
            if not ids:
                continue
            for oid in list(ids):
                self.CancelOrder(oid)
            self.openOrderIdsBySymbol[s] = []

    def IsQuarterStart(self, time: datetime) -> bool:
        is_first_day = time.day == 1
        is_quarter_month = time.month in [1, 4, 7, 10]
        return is_first_day and is_quarter_month

    def AllReadyForCovariance(self) -> bool:
        for s in self.symbols:
            if len(self.returns[s]) < self.lookbackDays:
                return False
        return True

    def BuildReturnMatrix(self, assets):
        T = self.lookbackDays
        N = len(assets)
        mat = np.zeros((T, N), dtype=float)
        for j, s in enumerate(assets):
            r = list(self.returns[s])
            if len(r) < T:
                return None
            mat[:, j] = np.array(r[-T:], dtype=float)
        return mat

    def CovToCorr(self, cov: np.ndarray) -> np.ndarray:
        n = cov.shape[0]
        corr = np.zeros((n, n), dtype=float)
        diag = np.diag(cov)
        std = np.sqrt(np.maximum(diag, 1e-18))
        for i in range(n):
            for j in range(n):
                denom = std[i] * std[j]
                if denom > 0:
                    corr[i, j] = cov[i, j] / denom
                else:
                    corr[i, j] = 0.0
        corr = np.clip(corr, -1.0, 1.0)
        return corr

    def SingleLinkageOrder(self, assets, dist):
        n = len(assets)
        if n <= 1:
            return list(assets)

        clusters = {i: [i] for i in range(n)}
        active = set(clusters.keys())
        next_id = n
        merges = {}

        def cluster_distance(a_id, b_id):
            a = clusters[a_id]
            b = clusters[b_id]
            m = float('inf')
            for i in a:
                for j in b:
                    d = float(dist[i, j])
                    if d < m:
                        m = d
            return m

        while len(active) > 1:
            active_list = list(active)
            best = float('inf')
            best_pair = None
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    ai = active_list[i]
                    bj = active_list[j]
                    d = cluster_distance(ai, bj)
                    if d < best:
                        best = d
                        best_pair = (ai, bj)

            if best_pair is None:
                return None

            left, right = best_pair
            clusters[next_id] = clusters[left] + clusters[right]
            merges[next_id] = (left, right)

            active.remove(left)
            active.remove(right)
            active.add(next_id)
            next_id += 1

        root = list(active)[0]

        def leaf_order(node_id):
            if node_id < n:
                return [node_id]
            left, right = merges[node_id]
            return leaf_order(left) + leaf_order(right)

        idx_order = leaf_order(root)
        return [assets[i] for i in idx_order]

    def HRPWeights(self, ordered_assets, cov_full, full_assets):
        index = {s: i for i, s in enumerate(full_assets)}

        L = [list(ordered_assets)]
        w = {s: 1.0 for s in ordered_assets}

        while True:
            big = [c for c in L if len(c) > 1]
            if len(big) == 0:
                break

            new_L = []
            for cluster in L:
                if len(cluster) == 1:
                    new_L.append(cluster)
                    continue

                m = len(cluster)
                split = m // 2
                left = cluster[:split]
                right = cluster[split:]
                children = [left, right]

                variances = []
                for subset in children:
                    V = self.SubCov(cov_full, subset, index)
                    if V is None:
                        return None
                    v = self.ClusterVarianceIVP(V)
                    v = max(float(v), 1e-18)
                    variances.append(float(v))

                inv_vars = [1.0 / v for v in variances]
                inv_sum = sum(inv_vars)
                if inv_sum <= 0:
                    return None
                alpha = [iv / inv_sum for iv in inv_vars]

                constrained_alpha = self.ConstrainSplit(cluster, children, alpha)
                if constrained_alpha is None:
                    return None

                for j, subset in enumerate(children):
                    a = float(constrained_alpha[j])
                    for s in subset:
                        w[s] *= a

                new_L.extend(children)

            L = new_L

        total = sum(w.values())
        if total <= 0:
            return None
        for s in w:
            w[s] = float(w[s] / total)
        return w

    def ClusterVarianceIVP(self, V: np.ndarray) -> float:
        diag = np.diag(V)
        diag = np.maximum(diag, 1e-18)
        iv = 1.0 / diag
        s = float(np.sum(iv))
        if s <= 0:
            m = len(diag)
            ww = np.ones((m, 1), dtype=float) / float(m)
            v = float((ww.T @ V @ ww)[0, 0])
            return float(v)
        ww = (iv / s).reshape(-1, 1)
        v = float((ww.T @ V @ ww)[0, 0])
        return float(v)

    def SubCov(self, cov_full, subset, index_map):
        idx = [index_map[s] for s in subset]
        V = cov_full[np.ix_(idx, idx)]
        return V

    def ConstrainSplit(self, parent_cluster, children, alpha):
        asset_alpha = {}
        for j, subset in enumerate(children):
            if len(subset) == 0:
                return None
            per_asset = float(alpha[j]) / float(len(subset))
            for s in subset:
                asset_alpha[s] = per_asset

        group_assets = {}
        for s in parent_cluster:
            g = self.groupId[s]
            if g not in group_assets:
                group_assets[g] = []
            group_assets[g].append(s)

        asset_max = {}
        asset_min = {}
        for s in parent_cluster:
            g = self.groupId[s]
            ga = group_assets[g]

            sum_alpha_group = sum(float(asset_alpha[x]) for x in ga)
            gmax = float(self.groupMax.get(g, 1.0))
            gmin = float(self.groupMin.get(g, 0.0))
            ai = float(asset_alpha[s])

            if sum_alpha_group <= 0:
                asset_max[s] = float(self.assetMax.get(s, 1.0))
                asset_min[s] = float(self.assetMin.get(s, 0.0))
                continue

            # Treat groupMax/groupMin as TOTAL group caps (do not multiply by group size)
            amax_i = ai * (gmax / sum_alpha_group) if gmax >= 0 else float(self.assetMax.get(s, 1.0))

            if gmin > 0:
                amin_i = ai * (sum_alpha_group / gmin)
            else:
                amin_i = float(self.assetMin.get(s, 0.0))

            asset_max[s] = min(float(self.assetMax.get(s, 1.0)), float(amax_i))
            asset_min[s] = max(float(self.assetMin.get(s, 0.0)), float(amin_i))

        subset_min = []
        subset_max = []
        for subset in children:
            subset_min.append(sum(float(asset_min[s]) for s in subset))
            subset_max.append(sum(float(asset_max[s]) for s in subset))

        constrained = self.ClampAndRenormalize(alpha, subset_min, subset_max)
        return constrained

    def ClampAndRenormalize(self, alpha, amin, amax):
        J = len(alpha)
        a = [float(x) for x in alpha]

        t = []
        for j in range(J):
            lo = float(amin[j])
            hi = float(amax[j])
            x = float(a[j])
            if x > hi:
                x = hi
            if x < lo:
                x = lo
            t.append(x)

        for _ in range(50):
            ssum = sum(t)
            diff = 1.0 - ssum
            if abs(diff) < 1e-10:
                break

            k = []
            for j in range(J):
                if abs(t[j] - a[j]) < 1e-12:
                    k.append(j)

            if len(k) == 0:
                if ssum > 0:
                    t = [x / ssum for x in t]
                break

            denom = sum(a[j] for j in k)
            if denom <= 0:
                break

            for j in k:
                t[j] = t[j] + abs(diff) * a[j] / denom

            for j in range(J):
                lo = float(amin[j])
                hi = float(amax[j])
                if t[j] > hi:
                    t[j] = hi
                if t[j] < lo:
                    t[j] = lo

        ssum = sum(t)
        if ssum > 0:
            t = [x / ssum for x in t]
        return t

    def RebalanceWithLimitOrders(self, weights):
        total_value = float(self.Portfolio.TotalPortfolioValue)

        for s in self.symbols:
            if s not in weights:
                continue
            if not self.Securities.ContainsKey(s):
                continue

            price = float(self.Securities[s].Price)
            if price <= 0:
                continue

            target_value = total_value * float(weights[s])
            target_qty = target_value / price

            current_qty = float(self.Portfolio[s].Quantity)
            delta_qty = target_qty - current_qty

            if abs(delta_qty) < 1.0:
                continue

            for oid in list(self.openOrderIdsBySymbol.get(s, [])):
                self.CancelOrder(oid)
            self.openOrderIdsBySymbol[s] = []

            qty = int(delta_qty)
            if qty == 0:
                continue

            order_id = self.LimitOrder(s, qty, price)
            self.openOrderIdsBySymbol[s].append(order_id)