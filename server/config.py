from ta import *
from utils import *

class config:
    
    # Runtime
    cache_dir = '__pycache__/'
    db_recalc_wind = 100

    # Data
    data_dir = 'data/histdata/'
    currencies = ['AUDJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'USDMXN']
    slice_hist_data = 0.1
    
    # Trading Strategy
    input_calculators = [trend.dpo, trend.macd, trend.macd_signal, trend.macd_diff, momentum.tsi, momentum.rsi, trend.trix, volatility.bollinger_hband, volatility.bollinger_lband]
    output_calculators = [lambda s : calcSmoothedGains(s, 30, 6*60)]
    train_learner = trainDecisionTree
    
    # API
    api_hostname = 'api-fxpractice.oanda.com'
    stream_hostname = 'stream-fxpractice.oanda.com'
    port = 443
    account_id = '101-002-9437682-001'
    token = '5475d1e27400cbbf2c48155ceaf97ba0-10d9868923e692b5c605805070f77b52'
