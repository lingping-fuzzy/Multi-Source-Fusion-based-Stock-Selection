import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
import os
from dtw_nofuture import dtw_measure
from cost_function import load_data_yahoo

# this is the name_base when we save the prediction results based on each windowsize parameter.
# for example, 5_sl_org__100.pkl--> we use windowsize(5), and use reference objective function.
# 5_sl_plus_01__100.pkl --> we use windowsize(5), and use  proposed objective function.
dataname = ['_sl_org__100.pkl',
            '_sl_plus_01__100.pkl']

# just a intermediate names used in codes,
maename = ['square',
           'sl_plus_01__100']

# suppose we have all results based on all windowsize parameters, and save in the loacation below.
locat = '\\dtwfinance\\'


# when we know which parameter is good, now we get the relevant results form
# the data with filename as-> '(window size paramter)_sl_plus_01__100.pkl'
############################# very important###############
# so now, when we know the exactly the window-size parameter, we should get the relevant results from that data
# we have len(Dase)--stocks, and each stock has its own window-size parameter, for example, Dase[0]=5, which means
# the first stock, its best windowsize is '5'., then as following '42' is for the second stock.
Dase = [5, 42, 15, 15, 15, 42, 42,  42, 42, 42, 42,  42, 5, 5, 42,  42, 42, 42]


# get the monthly_weekly results from corresponding window-size paramters results for each stock.
def get_monthly_weekly():
    stock_yahoofinance, stocks_yahoofinance_edited = load_data_yahoo()
    com_num = len(stocks_yahoofinance_edited.index)

    weekly =[]
    monthly=[]
    for col, item in enumerate(stocks_yahoofinance_edited.index):
        dId = Dase[col]
        weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data(base=str(dId), file=dataname[1])
        weekly.append(weekly_xgb[col])
        monthly.append(monthly_xgb[col])
    print(weekly)
    return  weekly, monthly


### the following code are actually the similar as file in 'compare_Windowsize.py'
def read_data(base=None, file=None):

    txtfile = open((locat + base + file), 'rb')
    resu = pickle.load(txtfile)
    txtfile.close()
    weekly_xgb2 = resu['weekly_xgb0']
    monthly_xgb2 = resu['monthly_xgb0']
    MAEall_xgb2 = resu['MAEall_xgb0']
    preds_xgb2 = resu['preds_xgb0']
    trueYs_xgb2 = resu['trueYs_xgb0']
    return weekly_xgb2, monthly_xgb2, MAEall_xgb2, preds_xgb2, trueYs_xgb2

def individual_MAE(item, preds_, data=None, xlen=50):
    x = preds_.squeeze()
    y = data[item]
    Mae_error = mean_absolute_error(x[-xlen:], y[-xlen:])
    MSLE_error = mean_squared_log_error(x[-xlen:], y[-xlen:])
    return Mae_error, MSLE_error

def dtw_distance_(data, preds_, n_out=27):

    dist = np.zeros(len(data))
    td = int(n_out/2)
    for i in range(0, len(data) - n_out, td):
        s1 = np.array(preds_[i:(i + n_out)])
        s2 = np.array(data[i:(i + n_out)])
        DD, PATH, distance = dtw_measure(s1, s2, w=5)
        dist[i: i + td] = distance/n_out  # *0.1
    return dist

def dtw_distance_individual(item, data, preds_):
    x = preds_[-50:]
    y = data[item][-50:]
    dt = dtw_distance_(x, y)
    dist = dt.sum() / 50
    return dist

def get_distmae_individual():
    stock_yahoofinance, stocks_yahoofinance_edited = load_data_yahoo()
    com_num = len(stocks_yahoofinance_edited.index)
    globals()['Mae0'] =[]
    globals()['Mae1'] = []
    globals()['RMSLE0'] =[]
    globals()['RMSLE1'] = []
    globals()['dist0'] =[]
    globals()['dist1'] = []
    globals()['Total'] =[]
    for col, item in enumerate(stocks_yahoofinance_edited.index):
        dId = Dase[col]
        for run in range(len(dataname)):
            weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data(base=str(dId), file=dataname[run])
            # weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data( file=dataname[run])
            mae, MSLE_error = individual_MAE(item, preds_xgb[item], data=stock_yahoofinance)
            var = 'Mae'+str(run)
            globals()[var].append(mae)
            var = 'RMSLE' + str(run)
            globals()[var].append(MSLE_error)
            dist = dtw_distance_individual(item, stock_yahoofinance, preds_xgb[item])
            var = 'dist'+str(run)
            globals()[var].append(dist)
    d_mae = np.concatenate((np.reshape(globals()['Mae0'], (1, com_num)).T), axis=0)
    d_mae = np.concatenate((d_mae.reshape(1, com_num), np.array(globals()['Mae1']).reshape(1, com_num)), axis=0)
    d_mae = np.concatenate((d_mae.reshape(2, com_num), np.array(globals()['dist0']).reshape(1, com_num)), axis=0)
    d_mae = np.concatenate((d_mae.reshape(3, com_num), np.array(globals()['dist1']).reshape(1, com_num)), axis=0)

    total2 = np.array(globals()['Mae1']) + np.array(globals()['dist1'])
    total1 = np.array(globals()['Mae0']) + np.array(globals()['dist0'])
    d_mae = np.concatenate((d_mae.reshape(4, com_num), total1.reshape(1, com_num)), axis = 0)
    d_mae = np.concatenate((d_mae.reshape(5, com_num), total2.reshape(1, com_num)), axis = 0)

    d_mae = np.concatenate((d_mae.reshape(6, com_num), np.array(globals()['RMSLE0']).reshape(1, com_num)), axis = 0)
    d_mae = np.concatenate((d_mae.reshape(7, com_num), np.array(globals()['RMSLE1']).reshape(1, com_num)), axis = 0)

    df_mae = pd.DataFrame(d_mae, columns=stocks_yahoofinance_edited.index[:com_num])
    return df_mae

def get_Table_individual():
    stock_yahoofinance, stocks_yahoofinance_edited = load_data_yahoo()
    com_num = len(stocks_yahoofinance_edited.index)

    globals()['Mae0'] =[]
    globals()['Mae1'] = []
    globals()['RMSLE0'] =[]
    globals()['RMSLE1'] = []
    globals()['dist0'] =[]
    globals()['dist1'] = []
    globals()['Total'] =[]
    for col, item in enumerate(stocks_yahoofinance_edited.index):
        dId = Dase[col]
        for run in range(len(dataname)):
            weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data(base=str(dId), file=dataname[run])
            # weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data( file=dataname[run])
            mae, MSLE_error = individual_MAE(item, preds_xgb[item], data=stock_yahoofinance)
            var = 'Mae'+str(run)
            globals()[var].append(mae)
            var = 'RMSLE' + str(run)
            globals()[var].append(MSLE_error)
            dist = dtw_distance_individual(item, stock_yahoofinance, preds_xgb[item])
            var = 'dist'+str(run)
            globals()[var].append(dist)
    d_mae = np.concatenate((np.reshape(globals()['Mae0'], (1, com_num)).T), axis=0)
    d_mae = np.concatenate((d_mae.reshape(1, com_num), np.array(globals()['Mae1']).reshape(1, com_num)), axis=0)
    d_mae = np.concatenate((d_mae.reshape(2, com_num), np.array(globals()['dist0']).reshape(1, com_num)), axis=0)
    d_mae = np.concatenate((d_mae.reshape(3, com_num), np.array(globals()['dist1']).reshape(1, com_num)), axis=0)

    total2 = np.array(globals()['Mae1']) + np.array(globals()['dist1'])
    total1 = np.array(globals()['Mae0']) + np.array(globals()['dist0'])
    d_mae = np.concatenate((d_mae.reshape(4, com_num), total1.reshape(1, com_num)), axis = 0)
    d_mae = np.concatenate((d_mae.reshape(5, com_num), total2.reshape(1, com_num)), axis = 0)

    d_mae = np.concatenate((d_mae.reshape(6, com_num), np.array(globals()['RMSLE0']).reshape(1, com_num)), axis = 0)
    d_mae = np.concatenate((d_mae.reshape(7, com_num), np.array(globals()['RMSLE1']).reshape(1, com_num)), axis = 0)

    df_mae = pd.DataFrame(d_mae, columns=stocks_yahoofinance_edited.index[:com_num])
    return df_mae

mae = get_Table_individual()  # this is obtain a framedata table with all evaluated metric values.
print(mae)
