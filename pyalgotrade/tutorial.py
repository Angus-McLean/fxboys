from __future__ import print_function

from pyalgotrade import strategy
from pyalgotrade import technical
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma
from pyalgotrade.technical import bollinger
import indicators

import pandas as pd

df = pd.DataFrame()

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, smaPeriod):
        super(MyStrategy, self).__init__(feed, 1000)
        self.__position = None
        self.__instrument = instrument
        # We'll use adjusted close values instead of regular close values.
        self.setUseAdjustedValues(True)

        ## Add calculators
        self.__calculators = {
            'sma' : ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod),
            'bol' : bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), smaPeriod, 1),
            'ta_dpo' : technical.EventBasedFilter(feed[instrument].getPriceDataSeries(), indicators.Ta_dpo(40)),
            'ta_tsi' : technical.EventBasedFilter(feed[instrument].getPriceDataSeries(), indicators.Ta_tsi(40)),
            'ta_trix' : technical.EventBasedFilter(feed[instrument].getPriceDataSeries(), indicators.Ta_trix(40))
        }
        
        ## Add indicators
        self.__indicators = {
            'sma' : lambda : self._MyStrategy__calculators['sma'][-1],
            'bol_low' : lambda : self._MyStrategy__calculators['bol'].getLowerBand()[-1],
            'bol_mid' : lambda : self._MyStrategy__calculators['bol'].getMiddleBand()[-1],
            'bol_high' : lambda : self._MyStrategy__calculators['bol'].getUpperBand()[-1],
            'ta_dpo' : lambda : self._MyStrategy__calculators['ta_dpo'][-1],
            'ta_tsi' : lambda : self._MyStrategy__calculators['ta_dpo'][-1],
            'ta_trix' : lambda : self._MyStrategy__calculators['ta_dpo'][-1]
        }

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def getIndicators(self):
        dictInds = {}
        for key in self._MyStrategy__indicators.keys():
            dictInds[key] = self._MyStrategy__indicators[key]()
        return dictInds

    def create_row(self, bar):
        indDict = self.getIndicators()
        barVals = list(bar.__getstate__())
        indVals = list(indDict.values())

        rowValues = barVals + indVals
        rowKeys = list(bar.__slots__) + list(indDict.keys())

        return dict(zip(rowKeys, rowValues))

    def onBars(self, bars):
        bar = bars[self.__instrument]

        rowObj = self.create_row(bar)
        rowDf  = pd.DataFrame([rowObj], columns=rowObj.keys())
        global df
        df = df.append(rowDf)
        
        # Wait for enough bars to be available to calculate a SMA.
        if self.__calculators['sma'][-1] is None:
            return
        
        # If a position was not opened, check if we should enter a long position.
        if self.__position is None:
            if bar.getPrice() > self.__calculators['sma'][-1]:
                # Enter a buy market order for 10 shares. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, 10, True)
        # Check if we have to exit the position.
        elif bar.getPrice() < self.__calculators['sma'][-1] and not self.__position.exitActive():
            self.__position.exitMarket()


def run_strategy(smaPeriod):
    # Load the bar feed from the CSV file
    feed = quandlfeed.Feed()
    feed.addBarsFromCSV("orcl", "pyalgotrade/WIKI-ORCL-2000-quandl.csv")

    # Evaluate the strategy with the feed.
    myStrategy = MyStrategy(feed, "orcl", smaPeriod)
    myStrategy.run()
    print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())

run_strategy(15)

print(df)