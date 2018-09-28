# NOTE : Run file from root of project

# imports
from ta import *
import pandas, numpy as np
import pickle
from utils import *
from config import config

# Ghetto Import
# execfile('./utils.py')
# execfile('./server/config.py')


# Read Datas
dataframes = [readAllDatForCurrency(config.data_dir, cur) for cur in config.currencies]
if config.slice_hist_data : 
    for df in dataframes:
        slice_ind = int(len(df)*config.slice_hist_data)
        df = df[-slice_ind:]

# Add input and output rows to histdata dfs
allDf = pd.DataFrame()
for df in dataframes:
    df_inds = runCalculators(df.close, config.input_calculators)
    df_outs = runCalculators(df.close, config.output_calculators)
    df = pd.concat([df, df_inds, df_outs], axis=1, join_axes=[df.index])
    df = df.dropna()
    allDf = pd.concat([allDf, df])

# Concat dataframes and train to learner
df_inds = allDf[df_inds.columns]
df_outs = allDf[df_outs.columns]

learner = trainDecisionTree(df_inds, df_outs)
pickle.dump(learner, open(config.cache_dir+'learner.sav', 'wb'))

print('Saved Learner to ', config.cache_dir+'learner.sav')