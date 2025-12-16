from AlgorithmImports import *

class MyAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.SetWarmUp(timedelta(days=30))

        self.symbols = []
        tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE']
        for ticker in tickers:
            # Fix: use Minute resolution so OnData (and exit logic) runs every minute
            equity = self.AddEquity(ticker, Resolution.Minute)
            self.symbols.append(equity.Symbol)

        self.holdingStartDate = {}
        self.openOrderTickets = {}   # symbol -> list(orderId)
        self.orderAges = {}          # orderId -> time submitted

        self.RecoverState()

    def RecoverState(self):
        self.bmCutoffsByFY = {}
        self.rebalanceSchedule = {}
        self.universeHighBM = set()
        self.fScore = {}
        self.targetPortfolioSymbols = set()
        self.nextRebalanceDate = None

        for symbol in self.symbols:
            p = self.Portfolio[symbol]
            if p.Invested and symbol not in self.holdingStartDate:
                self.holdingStartDate[symbol] = self.Time.date()

            if symbol not in self.openOrderTickets:
                self.openOrderTickets[symbol] = []

    def OnData(self, data: Slice):
        today = self.Time.date()

        # Active guard: always run exit logic (must run every minute; ensured by Minute subscriptions)
        self.ManageHoldingWindowExits(today, data)

        # Operational cleanup placeholder (cannot cancel without Transactions API in reference)
        self.CancelStaleLimitOrders(today, max_age_days=3)

        # No fundamentals API is available in the provided reference, so we cannot implement:
        # - Common-stock universe construction
        # - Book-to-market quintiles (BM)
        # - Piotroski F-SCORE (9 signals)
        # - Fiscal-year timing (formation = start of 5th month after FYE)

        # Do not place orders during warmup (but still allow the method to run for exits/guards)
        if self.IsWarmingUp:
            return

        # No further action possible under strict API constraints.

    def ManageHoldingWindowExits(self, today, data: Slice):
        # Close positions after 12 months using LimitOrders at current close
        for symbol in self.symbols:
            p = self.Portfolio[symbol]
            if not p.Invested:
                continue

            start_date = self.holdingStartDate.get(symbol, None)
            if start_date is None:
                continue

            if today < (start_date + timedelta(days=365)):
                continue

            if not data.Bars.ContainsKey(symbol):
                continue

            price = data.Bars[symbol].Close
            qty_to_close = -p.Quantity
            if qty_to_close == 0:
                continue

            ticket = self.LimitOrder(symbol, qty_to_close, price)
            if ticket is not None:
                oid = ticket.OrderId
                if symbol not in self.openOrderTickets:
                    self.openOrderTickets[symbol] = []
                self.openOrderTickets[symbol].append(oid)
                self.orderAges[oid] = self.Time

            # Prevent repeated submissions; without order-event APIs we cannot confirm fills here
            if symbol in self.holdingStartDate:
                del self.holdingStartDate[symbol]

    def CancelStaleLimitOrders(self, today, max_age_days=3):
        # The reference does not include Transactions APIs (e.g., GetOpenOrders/CancelOrder),
        # so we cannot cancel orders without hallucinating.
        return