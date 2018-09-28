
import v20
import pandas, numpy as np
import pickle
from config import config
import utils
from dateutil.parser import parse


db = {}
globals = {
    'learner' : None
}

def init():
    
    # init api
    print('Init API Connection')
    api = init_api()

    # Load learner
    print('Loading Pretrained Learner')
    globals['learner'] = pickle.load(open(config.cache_dir+'learner.sav', 'rb'))

    # Populate db
    print('Reading + Recalculating Initial Data')
    init_db()

    # Start polling
    print('Start Polling for Prices')
    connect_stream(api)

def init_api():
    api = v20.Context(
        config.stream_hostname,
        config.port,
        token=config.token
    )
    return api


def init_db():
    for cur in config.currencies:
        db[cur] = utils.readAllDatForCurrency(config.data_dir, cur)[-config.db_recalc_wind:]
        db[cur] = recalc_inds(db[cur])

        tmpDf = db[cur].dropna()
        preds = tmpDf.apply(calc_pred, axis=1)
        db[cur] = pandas.concat([db[cur], preds], axis=1, join_axes=[db[cur].index])



def start_poll():
    response = api.pricing.get(
        account_id,
        instruments=",".join(args.instrument),
        since=latest_price_time,
        includeUnitsAvailable=False
    )

    for price in response.get("prices", 200):
        if latest_price_time is None or price.time > latest_price_time:
            print(price_to_string(price))

    for price in response.get("prices", 200):
        if latest_price_time is None or price.time > latest_price_time:
            latest_price_time = price.time

    return latest_price_time


def connect_stream(api):
    instruments = [cur[:3]+'/'+cur[-3:] for cur in config.currencies]
    response = api.pricing.stream(
        config.account_id,
        snapshot=False,
        instruments=",".join(instruments),
    )
    
    #
    # Print out each price as it is received
    #
    for msg_type, msg in response.parts():
        if msg_type == "pricing.Price":
           proc_update(msg)


def recalc_inds(df):
    '''
    Rerun input_calculators
    '''
    df_inds = utils.runCalculators(df.close, config.input_calculators)
    df = pandas.concat([df, df_inds], axis=1, join_axes=[df.index])
    return df


def proc_update(msg):
    print('proc_update :', msg)
    if not msg.instrument:
        return
    cur = msg.instrument.replace('_', '')
    row = oanda_msg_to_row(msg)
    # https://stackoverflow.com/questions/50840769/insert-row-with-datetime-index-to-dataframe

    db[cur].append(pandas.DataFrame(row, columns=db[cur].columns ,index=[row['datetime']]))
    tmpDb = recalc_inds(db[cur][-config.db_recalc_wind:])

    pred = calc_pred(db[cur].iloc[-1])
    print('PREDICTION : ', msg.instrument, pred)


def oanda_msg_to_row(msg):
    price = (msg.asks[0].price - msg.bids[0].price) / 2 + msg.bids[0].price
    rowObj = {
        'datetime' : parse(msg.time),
        'open' : price,
        'close' : price
    }
    return rowObj


def calc_pred(row):
    input_cols = [fn.__name__ for fn in utils.input_calculators]
    rowData = row[input_cols].values.reshape(1,-1)
    return globals['learner'].predict(rowData)


init()