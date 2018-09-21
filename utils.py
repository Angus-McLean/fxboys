# utils.py

from os import listdir
from ta import *
import pandas, numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


################################################
###### Reading Data
################################################

def readDEX(filepath):
    rawData = pandas.read_csv(filepath).replace('.', np.nan).fillna(method='ffill')
    # TODO : Parse dates
    rawData.VALUE = rawData.VALUE.astype(float)
    return rawData


def readDAT(filename):
    """
    Reads DAT csv file and returns the dataframe.
    These files can be downloaded from : http://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d')
    
    df = pandas.read_csv(filename, names=['date','time','open', 'max', 'min', 'close', 'vol'])
    df['datetime'] = df['date'] + ' - ' + df['time']
    df['datetime'] = pandas.to_datetime(df['datetime'], format='%Y.%m.%d - %H:%M')
    return df

def readAllDatForCurrency(data_dir, currencyCode):
    """
    Given a currency code and a directory, will read all DAT files for that currency code into one large dataframe.
    """
    dataFileNames = list(filter(lambda a : (currencyCode in a), listdir(data_dir)))
    dfs = [readDAT(data_dir + name) for name in dataFileNames]
    allDf = pd.concat(dfs)
    allDf = allDf.sort_values(by=['datetime'], ascending=True).reset_index(drop=True).set_index('datetime')
    return allDf
    

################################################
###### Processing Input/Output Cols
################################################

input_calculators=[trend.dpo, trend.macd, trend.macd_signal, trend.macd_diff, momentum.tsi, momentum.rsi, trend.trix, volatility.bollinger_hband, volatility.bollinger_lband]
output_calculators=[calcGain]

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


def trainDecisionTree(inputDf, outputDf):
    """
    Takes inputDf and outputDf and trains and returns a learner
    """
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(inputDf, outputDf)
    return clf


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
    cumGains = others.daily_return(series).cumsum().fillna(0)
    tradeGains = [cumGains[t[1]] - cumGains[t[0]] for t in trades]
    tradeGains = np.add(np.divide(tradeGains, 100), 1.0)
    return np.product(tradeGains)



def runDTBacktest(df, 
                  filename=None,
                  input_calculators=input_calculators,
                  output_calculators=output_calculators,
                  trainLearner=trainDecisionTree,
                  make_trades=makeTrades,
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
        'filename':filename,
        'gain' : totalGain,
        'dt' : decTree,
        'df':df,
        'preds':preds,
        'trades':trades,
        'test_df':test_df
    }