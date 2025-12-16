
class TestAlgo(QCAlgorithm):
    def Initialize(self):
        self.AddEquity("SPY", Resolution.Minute)
        self.SetFillModel(ImmediateFillModel())
    def OnData(self, data):
        self.SetHoldings("SPY", 1)
    def RecoverState(self): pass
