# lean_api.py

LEAN_API_CHEATSHEET = """
# =============================================================================
#  QUANTCONNECT LEAN API REFERENCE (LOCAL DATA SAFE MODE)
#  STRICT COMPLIANCE REQUIRED. DO NOT GUESS API METHODS.
# =============================================================================

# --- 1. IMPORTS ---
from AlgorithmImports import *

# --- 2. SETUP ---
class MyAlgo(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(START_YEAR, START_MONTH, START_DAY)
        self.SetEndDate(END_YEAR, END_MONTH, END_DAY)
        self.SetCash(STARTING_CASH)
        self.SetWarmUp(timedelta(days=30))
        
        self.symbols = []
        target_tickers = ["TICKER1"] 
        for ticker in target_tickers:
            equity = self.AddEquity(ticker, Resolution.Minute)
            self.symbols.append(equity.Symbol)

# --- 3. INDICATOR CREATION (Strict Signatures) ---
# NOTE: You MUST check `if not indicator.IsReady: return` before use.

# SIMPLE (Symbol, Period, Resolution)
# self.EMA(symbol, period, Resolution.Minute)
# self.SMA(symbol, period, Resolution.Minute)
# self.MAX(symbol, period, Resolution.Minute)
# self.MIN(symbol, period, Resolution.Minute)
# self.STD(symbol, period, Resolution.Minute)

# RSI & ATR (Must include MovingAverageType to prevent overload errors)
# self.RSI(symbol, period, MovingAverageType.Wilders, Resolution.Minute)
# self.ATR(symbol, period, MovingAverageType.Wilders, Resolution.Minute)

# ADX (Average Directional Index)
# self.ADX(symbol, period, Resolution.Minute)

# BOLLINGER BANDS
# self.BB(symbol, period, k_std, MovingAverageType.Simple, Resolution.Minute)

# MACD
# self.MACD(symbol, fast, slow, signal, MovingAverageType.Exponential, Resolution.Minute)

# AROON
# self.AROON(symbol, period, Resolution.Minute)

# --- 4. INDICATOR VALUE ACCESS (Critical!) ---
# Standard:
# value = indicator.Current.Value

# ADX Specific:
# adx_val = indicator.Current.Value
# p_di = indicator.PositiveDirectionalIndex.Current.Value
# n_di = indicator.NegativeDirectionalIndex.Current.Value

# Bollinger Bands Specific:
# upper = indicator.UpperBand.Current.Value
# middle = indicator.MiddleBand.Current.Value
# lower = indicator.LowerBand.Current.Value

# MACD Specific:
# macd_val = indicator.Current.Value      # The MACD line (Fast - Slow)
# signal = indicator.Signal.Current.Value # The Signal line
# hist = indicator.Histogram.Current.Value

# Aroon Specific:
# up = indicator.AroonUp.Current.Value
# down = indicator.AroonDown.Current.Value

# --- 5. DATA & HISTORY ---
    def OnData(self, data: Slice):
        if self.IsWarmingUp: return
        if not data.Bars.ContainsKey(symbol): return
        
        # Current Data (TradeBar)
        bar = data.Bars[symbol]
        price = bar.Close
        
        # History (Pandas DataFrame)
        # Returns: MultiIndex (symbol, time) or Index (time) depending on request
        # Columns are LOWERCASE: 'open', 'high', 'low', 'close', 'volume'
        
        # Example pattern for safety:
        # history = self.History(symbol, 20, Resolution.Minute)
        # if not history.empty:
        #     # Handle both single-index and multi-index safely
        #     if 'close' in history.columns:
        #         closes = history['close']
        #     else:
        #         closes = history.loc[symbol]['close']
        #     recent_high = history.iloc[-1]['high']

# --- 6. TRADING COMMANDS ---
# self.SetHoldings(symbol, 0.05)
# self.Liquidate(symbol)
# self.MarketOrder(symbol, 100)
# self.LimitOrder(symbol, 100, price)
# self.StopMarketOrder(symbol, -100, stop_price)

# --- 7. PORTFOLIO STATE ---
# p = self.Portfolio[symbol]
# p.Invested (bool)
# p.Quantity (float)
# p.AveragePrice (float)
# p.UnrealizedProfitPercent (float)
"""