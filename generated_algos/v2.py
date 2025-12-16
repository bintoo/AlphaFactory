from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import timedelta

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.SetWarmUp(timedelta(days=60))

        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            self.symbols.append(equity.Symbol)

        # Monthly scheduling state
        self.lastProcessedYear = -1
        self.lastProcessedMonth = -1
        self.lastRefitYear = -1

        # Model state
        self.theta = None
        self.feature_cols = None

        # Outputs/state (must be recovered)
        self.latestPred = {}
        self.targetWeights = {}

        # Strategy parameters
        self.valWindowYears = 2
        self.l2Grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

        self.RecoverState()

    def RecoverState(self):
        self.latestPred = {}
        self.targetWeights = {}

        for s in self.symbols:
            self.latestPred[s] = None
            self.targetWeights[s] = 0.0

            p = self.Portfolio[s]
            if p.Invested:
                price = self.Securities[s].Price
                if price > 0:
                    current_value = float(p.Quantity) * float(price)
                    self.targetWeights[s] = current_value / float(self.Portfolio.TotalPortfolioValue)

    def OnData(self, data: Slice):
        # Active guard: always call exit logic
        self.ExitLogic(data)

        # Only run monthly logic on first trading day of new month
        if not self._IsNewMonth():
            return

        panel = self._BuildMonthlyPanel_NoLookahead()
        if panel is None or panel.empty:
            return

        months = sorted(panel.index.get_level_values(0).unique())
        if len(months) < 6:
            return

        current_t = months[-1]

        xs_t = panel.xs(current_t, level=0)
        if xs_t is None or xs_t.empty:
            return

        # Annual refit (once per year)
        if (not self.IsWarmingUp) and self.Time.year != self.lastRefitYear:
            self._RefitModel(panel, current_t)
            self.lastRefitYear = self.Time.year

        if self.theta is None or self.feature_cols is None:
            return

        X_pred = xs_t.reset_index()
        if X_pred is None or len(X_pred) == 0:
            return

        # ensure required feature columns exist
        for c in self.feature_cols:
            if c not in X_pred.columns:
                return

        Xp = X_pred[self.feature_cols].astype(float).values
        Xp = self._AddIntercept(Xp)
        rhat = Xp.dot(self.theta)

        # Update latest predictions (Symbol keys, not strings)
        for i, row in enumerate(X_pred.itertuples(index=False)):
            sym = row.symbol
            if sym in self.latestPred:
                self.latestPred[sym] = float(rhat[i])

        # Place limit orders only when not warming up
        if not self.IsWarmingUp:
            self._SetDecileTargetsAndPlaceLimitOrders(data)

    def ExitLogic(self, data: Slice):
        pass

    def _IsNewMonth(self) -> bool:
        y = self.Time.year
        m = self.Time.month
        if y == self.lastProcessedYear and m == self.lastProcessedMonth:
            return False
        self.lastProcessedYear = y
        self.lastProcessedMonth = m
        return True

    def _BuildMonthlyPanel_NoLookahead(self):
        lookbackDays = 420  # ~20 months of trading days

        history = self.History(self.symbols, lookbackDays, Resolution.Daily)
        if history is None or history.empty:
            return None
        if not hasattr(history, "index"):
            return None

        # Build per-symbol monthly close and monthly return from history
        monthly_close_by_sym = {}
        monthly_ret_by_sym = {}

        # DATAFRAME SAFETY: isolate with .loc[sym]
        for sym in self.symbols:
            try:
                df = history.loc[sym].copy()
            except:
                continue

            if df is None or df.empty:
                continue
            if 'close' not in df.columns:
                continue

            df = df.dropna()
            if df.empty:
                continue

            df = df.sort_index()
            df["month"] = df.index.to_series().apply(lambda dt: pd.Timestamp(dt.year, dt.month, 1)).values

            monthly_close = df.groupby("month")["close"].last()
            monthly_ret = monthly_close.pct_change()

            monthly_close_by_sym[sym] = monthly_close
            monthly_ret_by_sym[sym] = monthly_ret

        if len(monthly_close_by_sym) == 0:
            return None

        macro = self._ComputeMonthlyMacro_FromMonthlyReturns(monthly_ret_by_sym)
        if macro is None or macro.empty:
            return None

        rows = []
        char_cols = ["mom_1m", "mom_3m", "mom_12m_ex1", "vol_1m", "vol_3m", "mdd_3m"]

        for sym in self.symbols:
            try:
                df = history.loc[sym].copy()
            except:
                continue

            if df is None or df.empty or 'close' not in df.columns:
                continue

            df = df.dropna()
            if df.empty:
                continue

            df = df.sort_index()
            df["ret_d"] = df["close"].pct_change()
            df["month"] = df.index.to_series().apply(lambda dt: pd.Timestamp(dt.year, dt.month, 1)).values

            if sym not in monthly_close_by_sym or sym not in monthly_ret_by_sym:
                continue

            monthly_close = monthly_close_by_sym[sym]
            monthly_ret = monthly_ret_by_sym[sym]

            months = list(monthly_close.index)
            if len(months) < 3:
                continue

            for i in range(1, len(months)):
                t = months[i]
                prev = months[i - 1]

                prev_month_df = df[df["month"] == prev]
                if prev_month_df.empty:
                    continue
                end_time = prev_month_df.index.max()

                trailing = df[df.index <= end_time]
                if trailing.empty:
                    continue

                mom_1m = self._WindowReturn(trailing["close"], 21)
                mom_3m = self._WindowReturn(trailing["close"], 63)
                mom_12m_ex1 = self._WindowReturnExcludeRecent(trailing["close"], 252, 21)
                vol_1m = self._WindowStd(trailing["ret_d"], 21)
                vol_3m = self._WindowStd(trailing["ret_d"], 63)
                mdd_3m = self._MaxDrawdown(trailing["close"], 63)

                fwd_ret = np.nan
                if t in monthly_ret.index and pd.notna(monthly_ret.loc[t]):
                    fwd_ret = float(monthly_ret.loc[t])

                rows.append({
                    "month": t,
                    "symbol": sym,
                    "fwd_ret": fwd_ret,
                    "mom_1m": mom_1m,
                    "mom_3m": mom_3m,
                    "mom_12m_ex1": mom_12m_ex1,
                    "vol_1m": vol_1m,
                    "vol_3m": vol_3m,
                    "mdd_3m": mdd_3m
                })

        if len(rows) == 0:
            return None

        panel = pd.DataFrame(rows)
        panel = panel.dropna(subset=["month", "symbol"])
        if panel.empty:
            return None

        panel = panel.merge(macro, on="month", how="left")
        # require all macro predictors present
        panel = panel.dropna(subset=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"])
        if panel.empty:
            return None

        # Cross-sectional impute + rank-map into [-1,1] per month
        for c in char_cols:
            panel[c] = panel.groupby("month")[c].transform(lambda s: self._FillWithMedian(s))
            panel[c + "_rk"] = panel.groupby("month")[c].transform(lambda s: self._RankMapMinusOneToOne(s))

        macro_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        rk_char_cols = [cc + "_rk" for cc in char_cols]

        # Interaction terms: x ⊗ c
        for xi in macro_cols:
            for cj in rk_char_cols:
                panel[f"{xi}__{cj}"] = panel[xi].astype(float) * panel[cj].astype(float)

        feature_cols = []
        feature_cols += macro_cols
        feature_cols += rk_char_cols
        feature_cols += [f"{xi}__{cj}" for xi in macro_cols for cj in rk_char_cols]

        panel = panel.set_index(["month", "symbol"]).sort_index()
        panel.attrs["feature_cols"] = feature_cols
        return panel

    def _ComputeMonthlyMacro_FromMonthlyReturns(self, monthly_ret_by_sym: dict):
        required = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "XLE"]

        ticker_to_symbol = {}
        for s in self.symbols:
            sec = self.Securities[s]
            if sec is None:
                continue
            ticker_to_symbol[sec.Symbol.Value] = s

        for t in required:
            if t not in ticker_to_symbol:
                return None
            if ticker_to_symbol[t] not in monthly_ret_by_sym:
                return None

        def mret(ticker):
            return monthly_ret_by_sym[ticker_to_symbol[ticker]]

        spy = mret("SPY")
        qqq = mret("QQQ")
        efa = mret("EFA")
        eem = mret("EEM")
        tlt = mret("TLT")
        ief = mret("IEF")
        gld = mret("GLD")
        vnq = mret("VNQ")
        xle = mret("XLE")

        idx = spy.index
        for s in [qqq, efa, eem, tlt, ief, gld, vnq, xle]:
            idx = idx.intersection(s.index)

        if len(idx) == 0:
            return None

        base = pd.DataFrame(index=idx)
        base["x1"] = spy.loc[idx]
        base["x2"] = tlt.loc[idx]
        base["x3"] = gld.loc[idx]
        base["x4"] = vnq.loc[idx]
        base["x5"] = qqq.loc[idx] - spy.loc[idx]
        base["x6"] = ief.loc[idx] - tlt.loc[idx]
        base["x7"] = eem.loc[idx] - efa.loc[idx]
        base["x8"] = xle.loc[idx] - spy.loc[idx]

        base = base.dropna()
        if base.empty:
            return None

        # shift macro forward by 1 month: x(t) = returns(t-1)
        base = base.reset_index().rename(columns={"index": "month"})
        base["month"] = base["month"].apply(lambda ts: pd.Timestamp(ts.year, ts.month, 1) + pd.offsets.MonthBegin(1))
        base = base.sort_values("month").reset_index(drop=True)
        return base

    def _RefitModel(self, panel: pd.DataFrame, current_t: pd.Timestamp):
        feature_cols = panel.attrs.get("feature_cols", None)
        if feature_cols is None or len(feature_cols) == 0:
            return

        all_months = sorted(panel.index.get_level_values(0).unique())
        past_months = [m for m in all_months if m < current_t]
        if len(past_months) < 18:
            return

        val_months_count = int(self.valWindowYears * 12)
        if len(past_months) <= val_months_count + 6:
            val_months = past_months[-max(6, min(val_months_count, len(past_months) // 2)):]
        else:
            val_months = past_months[-val_months_count:]

        train_months = [m for m in past_months if m < val_months[0]]
        if len(train_months) == 0:
            return

        train = panel.loc[(train_months, slice(None)), :].reset_index()
        val = panel.loc[(val_months, slice(None)), :].reset_index()

        train = train.dropna(subset=["fwd_ret"])
        val = val.dropna(subset=["fwd_ret"])
        if len(train) < 50 or len(val) < 20:
            return

        # verify columns
        for c in feature_cols:
            if c not in train.columns or c not in val.columns:
                return

        Xtr = train[feature_cols].astype(float).values
        ytr = train["fwd_ret"].astype(float).values
        Xva = val[feature_cols].astype(float).values
        yva = val["fwd_ret"].astype(float).values

        Xtr_i = self._AddIntercept(Xtr)
        Xva_i = self._AddIntercept(Xva)

        best_l2 = None
        best_mse = None
        best_theta = None

        for l2 in self.l2Grid:
            theta = self._RidgeFit(Xtr_i, ytr, l2)
            if theta is None:
                continue
            pred = Xva_i.dot(theta)
            mse = float(np.mean((yva - pred) ** 2))
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_l2 = l2
                best_theta = theta

        if best_theta is None or best_l2 is None:
            return

        final_theta = self._RidgeFit(Xtr_i, ytr, best_l2)
        if final_theta is None:
            return

        self.theta = final_theta
        self.feature_cols = feature_cols

    def _SetDecileTargetsAndPlaceLimitOrders(self, data: Slice):
        preds = []
        for s in self.symbols:
            p = self.latestPred.get(s, None)
            if p is None:
                continue
            if not data.Bars.ContainsKey(s):
                continue
            preds.append((s, p))

        if len(preds) < 10:
            return

        preds.sort(key=lambda x: x[1])
        n = len(preds)
        decile_size = max(1, n // 10)

        bottom = preds[:decile_size]
        top = preds[-decile_size:]

        long_weight_each = 0.5 / len(top)
        short_weight_each = -0.5 / len(bottom)

        for s in self.symbols:
            self.targetWeights[s] = 0.0
        for s, _ in top:
            self.targetWeights[s] = long_weight_each
        for s, _ in bottom:
            self.targetWeights[s] = short_weight_each

        # Place limit orders at last price snapshot
        for s in self.symbols:
            if not data.Bars.ContainsKey(s):
                continue

            bar = data.Bars[s]
            price = bar.Close
            if price <= 0:
                continue

            tgt_w = float(self.targetWeights.get(s, 0.0))
            tgt_value = tgt_w * float(self.Portfolio.TotalPortfolioValue)
            tgt_qty = int(tgt_value / float(price))

            cur_qty = int(self.Portfolio[s].Quantity)
            delta = tgt_qty - cur_qty
            if delta == 0:
                continue

            self.LimitOrder(s, delta, float(price))

    def _AddIntercept(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.concatenate([ones, X], axis=1)

    def _RidgeFit(self, X: np.ndarray, y: np.ndarray, l2: float):
        XtX = X.T.dot(X)
        p = XtX.shape[0]
        I = np.eye(p, dtype=float)
        I[0, 0] = 0.0
        A = XtX + float(l2) * I
        Xty = X.T.dot(y)
        try:
            theta = np.linalg.solve(A, Xty)
            return theta
        except:
            return None

    def _FillWithMedian(self, s: pd.Series) -> pd.Series:
        if s.isna().all():
            return s
        med = float(s.median(skipna=True))
        return s.fillna(med)

    def _RankMapMinusOneToOne(self, s: pd.Series) -> pd.Series:
        n = int(s.shape[0])
        if n <= 1:
            return pd.Series(np.zeros(n), index=s.index)
        ranks = s.rank(method="average", na_option="keep")
        u = (ranks - 0.5) / n
        mapped = 2.0 * u - 1.0
        mapped = mapped.fillna(0.0)
        return mapped

    def _WindowReturn(self, close: pd.Series, window: int):
        if close is None or close.empty or len(close) < window + 1:
            return np.nan
        c0 = float(close.iloc[-window - 1])
        c1 = float(close.iloc[-1])
        if c0 <= 0:
            return np.nan
        return (c1 / c0) - 1.0

    def _WindowReturnExcludeRecent(self, close: pd.Series, window: int, exclude: int):
        if close is None or close.empty:
            return np.nan
        if len(close) < window + exclude + 1:
            return np.nan
        end_idx = -exclude - 1
        start_idx = -exclude - window - 1
        c0 = float(close.iloc[start_idx])
        c1 = float(close.iloc[end_idx])
        if c0 <= 0:
            return np.nan
        return (c1 / c0) - 1.0

    def _WindowStd(self, rets: pd.Series, window: int):
        if rets is None or rets.empty or len(rets) < window:
            return np.nan
        w = rets.iloc[-window:]
        return float(w.std(ddof=1)) if w.dropna().shape[0] >= 2 else np.nan

    def _MaxDrawdown(self, close: pd.Series, window: int):
        if close is None or close.empty or len(close) < window:
            return np.nan
        w = close.iloc[-window:].astype(float)
        if w.isna().any():
            w = w.dropna()
        if w.empty:
            return np.nan
        running_max = w.cummax()
        dd = (w / running_max) - 1.0
        return float(dd.min()) if len(dd) > 0 else np.nan