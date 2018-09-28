# utils.py

from os import listdir
from ta import *
import re
import pandas, numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


################################################
###### Reading Data
################################################

def listCurrenciesInDir(data_dir):
    return list(np.unique(
        [re.sub('DAT_MT_(\w{6}).*', r'\1', a) for a in listdir(data_dir)]
        ))

def readDEX(filepath):
    rawData = pandas.read_csv(filepath).replace('.', np.nan).fillna(method='ffill')
    # TODO : Parse dates
    rawData.VALUE = rawData.VALUE.astype(float)
    return rawData


def readDAT(testname):
    """
    Reads DAT csv file and returns the dataframe.
    These files can be downloaded from : http://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d')
    
    df = pandas.read_csv(testname, names=['date','time','open', 'max', 'min', 'close', 'vol'])
    df['datetime'] = df['date'] + ' - ' + df['time']
    df['datetime'] = pandas.to_datetime(df['datetime'], format='%Y.%m.%d - %H:%M')
    return df

def readAllDatForCurrency(data_dir, currencyCode):
    """
    Given a currency code and a directory, will read all DAT files for that currency code into one large dataframe.
    """
    datatestnames = list(filter(lambda a : (currencyCode in a), listdir(data_dir)))
    dfs = [readDAT(data_dir + name) for name in datatestnames]
    allDf = pd.concat(dfs)
    allDf = allDf.sort_values(by=['datetime'], ascending=True).reset_index(drop=True).set_index('datetime')
    return allDf
    

################################################
###### Processing Input/Output Cols
################################################

def runCalculators(series, calculators=[]):
    """
    Takes series and runs all calculator functions and adds the columns to a DataFrame
    """
    df_obj = {}
    for ind, fn in enumerate(calculators):
        df_obj[fn.__name__] = fn(series)
    return pd.DataFrame(df_obj)

def calcSmoothedGains(series, shortWind, longWind):
    """
    Used for calculating an output column.
    Takes prices series and calculates 2 moving averages and compares them to identify up/down trends.
    """
    shortMa = series.rolling(str(shortWind)+'min').mean().shift(-shortWind)
    longMa = series.rolling(str(longWind)+'min').mean().shift(-longWind)

    # Calc Buy hold and Sell signals
    buySellRatio = longMa - shortMa
    return buySellRatio


################################################
###### Training Methods
################################################
def splitData(df, split):
    """
    Slices a dataframe into train and test dataframes based on split ratio
    Example arg: split=0.85
    Returns train_df, test_df
    """
    train = df.iloc[:int(len(df)*split)]
    test = df.iloc[int(len(df)*split):]
    
    return train, test


def walkForwardInds(df, num_tests, train_split):
    sampLen = len(df) / (train_split + num_tests*(1.0 - train_split))
    sampLen = int(sampLen)
    test_len = int(sampLen * (1.-train_split))
    
    splits = []
    for i in range(num_tests):
        start = test_len*i
        end = start + sampLen
        splits.append((start, end))
    return splits

def trainDecisionTree(inputDf, outputDf):
    """
    Takes inputDf and outputDf and trains and returns a learner
    """
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(inputDf, outputDf)
    return clf


################################################
###### Execute Trades
################################################
def stdConfidenceTrades(predictions, buy_confidence=1.5, sell_confidence=1.1):
    """
    Buy signals when prediction is above buy_confidence * standardDeviation and sells with similar metric
    """
    smooth_preds = pd.Series(predictions).rolling(5).mean()
    buy_thresh = np.mean(smooth_preds) + buy_confidence * np.std(smooth_preds)
    sell_thresh = np.mean(smooth_preds) - sell_confidence * np.std(smooth_preds)
    buy_positions = np.where(predictions > buy_thresh)[0]
    sell_positions = np.where(predictions < sell_thresh)[0]
    
    buys = buy_positions
    sells = []
    curSell = 0
    for curBuy in buys:
        arr = np.where(sell_positions > curBuy)[0]
        if len(arr):
            sells.append(sell_positions[arr[0]])
    tradePairs = list(zip(buys, sells))
    return tradePairs


################################################
###### Evaluate Performance
################################################
def getBuySellGains(series, trades):
    """
    Calculates the total gain of all trades made on a series of values

    Arguments : 
    - series : np.Series(priceTicksForStock)
    - trades : Array of tuples of (buyPosition, sellPosition), where positions are integer indexes to place trades
    """
    marketAlphas = others.daily_return(series)
    tradeGains = [
        np.product(np.add(np.divide(marketAlphas[t[0]:t[1]], 100), 1.0))
        for t in trades]
    return np.product(tradeGains)

def getSummary(backTestResult):
    
    returns = np.add(np.divide(others.daily_return(backTestResult['test_df'].close),100),1)
    
    return {
        'testname' : backTestResult['testname'],
        'algo_gain' : backTestResult['gain'],
        'buyhold_gain' : np.product(returns),
        'beat_market' : backTestResult['gain'] > np.product(returns),
        'start_date' : backTestResult['test_df'].index[0],
        'end_date' : backTestResult['test_df'].index[-1]
    }



################################################
###### Test Runners
################################################

input_calculators=[trend.dpo, trend.macd, trend.macd_signal, trend.macd_diff, momentum.tsi, momentum.rsi, trend.trix, volatility.bollinger_hband, volatility.bollinger_lband]
output_calculators=[lambda s : utils.calcSmoothedGains(s, 30, 6*60)]

def runDTBacktest(df, 
                  testname=None,
                  input_calculators=input_calculators,
                  output_calculators=output_calculators,
                  trainLearner=trainDecisionTree,
                  make_trades=stdConfidenceTrades,
                  calc_gains=getBuySellGains):
    
    """
    Takes a dataset and a number of functions and executes a simple 0.9 training split backtest
    Keyword arguments : total dataframe, input and output column calculators, learner, tradeDecision maker and evaluation function
    """
    df_inds = runCalculators(df.close, input_calculators)
    df_outs = runCalculators(df.close, output_calculators)
    df = pd.concat([df, df_inds, df_outs], axis=1, join_axes=[df.index])
    df = df.dropna()
    df_inds = df[df_inds.columns]
    df_outs = df[df_outs.columns]
    
    train_split = 0.9
    train_in, test_in = splitData(df_inds, train_split)
    train_out, test_out = splitData(df_outs, train_split)
    
    decTree = trainLearner(train_in, train_out)
    preds = decTree.predict(test_in)
    test_out['preds'] = preds
    trades = make_trades(preds)
    
    test_df = pd.concat([df, test_in, test_out], axis=1, join_axes=[test_in.index])
    totalGain = calc_gains(test_df['close'], trades)
    
    return {
        'testname':testname,
        'gain' : totalGain,
        'dt' : decTree,
        'df':df,
        'preds':preds,
        'trades':trades,
        'test_df':test_df
    }