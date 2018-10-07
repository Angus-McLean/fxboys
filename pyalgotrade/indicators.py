from __future__ import print_function

from pyalgotrade import dataseries
from pyalgotrade import technical

from ta import *
import pandas, numpy as np

input_calculators=[trend.dpo, trend.macd, trend.macd_signal, trend.macd_diff, momentum.tsi, momentum.rsi, trend.trix, volatility.bollinger_hband, volatility.bollinger_lband]

# class Ta_dpo(technical.EventWindow):
#     def getValue(self):
#         vals = trend.dpo(pandas.Series(self.getValues())).values
#         if len(vals) == 0:
#             return None
#         ret = vals[-1]
#         if math.isnan(ret) : 
#             return None
#         else :
#             return ret
#         # return ret if not math.isnan(ret) else None

class Ta_dpo(technical.EventWindow):
    def getValue(self):
        ret = trend.dpo(pandas.Series(self.getValues())).values[-1]
        return ret if not math.isnan(ret) else None


class Ta_tsi(technical.EventWindow):
    def getValue(self):
        ret = momentum.tsi(pandas.Series(self.getValues())).values[-1]
        return ret if not math.isnan(ret) else None

class Ta_trix(technical.EventWindow):
    def getValue(self):
        ret = trend.trix(pandas.Series(self.getValues())).values[-1]
        return ret if not math.isnan(ret) else None

