import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# suppose we use the windowsize parameters, and all the parameters are in Dbase
Dbase = [5, 7, 10, 12, 15, 17, 20, 25, 27, 30, 32, 35, 40, 42, 45, 52, 60, 63, 65]

# suppose we have all results based on all windowsize parameters, and save in the loacation below.
locat = '\\dtwfinance\\'

# function 'highlight_max' and  'highlightdf' are used in jupter notebook code, just to color the results
# for easily to recognize which results are best.
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')


def highlightdf(df):
    s2 = df.apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
    return s2



def read_data(base=None, file='org1.pkl'):
    txtfile = open(( locat+ base + file), 'rb')
    resu = pickle.load(txtfile)
    txtfile.close()
    weekly_xgb2 = resu['weekly_xgb0']
    monthly_xgb2 = resu['monthly_xgb0']
    MAEall_xgb2 = resu['MAEall_xgb0']
    preds_xgb2 = resu['preds_xgb0']
    trueYs_xgb2 = resu['trueYs_xgb0']
    return weekly_xgb2, monthly_xgb2, MAEall_xgb2, preds_xgb2, trueYs_xgb2

# function for calculate the metrics
def square_MAE(preds_, data=None, xlen=50):
    NAE = []
    NSE = []
    for item in preds_.keys():
        x = preds_[item].squeeze()
        y = data[item]
        Mae_error = mean_absolute_error(x[-xlen:], y[-xlen:])
        Msae_error = mean_squared_error(x[-xlen:], y[-xlen:])

        NAE.append(Mae_error)
        NSE.append(Msae_error)
    return NAE, NSE

# function for calculate the metrics
def dtw_distance_(data, preds_, n_out=27):
    dist = np.zeros(len(data))
    td = int(n_out / 2)
    for i in range(0, len(data) - n_out, td):
        s1 = np.array(preds_[i:(i + n_out)])
        s2 = np.array(data[i:(i + n_out)])
        DD, PATH, distance = dtw_measure(s1, s2, w=5)
        dist[i: i + td] = distance / n_out  # *0.1
    return dist

# function for calculate the metrics
def dtw_distance_cal(data, preds_):
    dist = np.zeros(len(preds_.keys()))
    i = 0
    for item in preds_.keys():
        # x = preds_[item].squeeze()
        # y = data[item]
        x = preds_[item][-50:]
        y = data[item][-50:]
        dt = dtw_distance_(x, y)
        dist[i] = dt.sum() / 50
        i = i + 1
    return dist

# function for remove the results by reference-based objective function, which is just used to
# easily to check the results (framedata results.)
def subtract_row(source, data, id1, id2):
    feature_list = source.columns
    zero_data = np.zeros(shape=(len(data), len(feature_list)))
    zd = pd.DataFrame(zero_data, columns=feature_list)
    zd.index = source.index
    zd.iloc[id1:id2 + 1] = source.iloc[id1]
    data = data - zd
    return data

# function used to call subtract_row
def basecompare(source):
    df = source.copy()
    for i in range(0, len(Dbase) * 2, 2):
        df = subtract_row(source, df, i, i + 1)
    return df


# to get the all results--just run this file, and the output is dataframe with all parameters and metric values.
def get_results_D():
    stock_yahoofinance, stocks_yahoofinance_edited = load_data_yahoo()
    com_num = len(stocks_yahoofinance_edited.index)

    indexname = []
    resindex = []
    for din in range(len(Dbase)):
        dId = Dbase[din]
        base = str(dId)
        for run in range(len(dataname)):  #
            var = 'preds_xgb' + str(run) + '-' + str(base)

            weekly_xgb, monthly_xgb, MAEall_xgb, preds_xgb, trueYs_xgb = read_data(base=base, file=dataname[run])
            globals()[var] = preds_xgb
            indexname.append((str(run) + '-' + str(base)))
        resindex.append((str(1) + '-' + str(base)))

        for run in range(len(maename)):
            var = 'mae' + maename[run] + '-' + str(base)
            var1 = 'mse' + maename[run] + '-' + str(base)
            varn = 'preds_xgb' + str(run) + '-' + str(base)
            mae, mse = square_MAE(globals()[varn], data=stock_yahoofinance)
            globals()[var] = mae
            globals()[var1] = mse

        for run in range(len(maename)):
            var = 'dist' + maename[run] + '-' + str(base)
            varn = 'preds_xgb' + str(run) + '-' + str(base)
            dist = dtw_distance_cal(stock_yahoofinance, globals()[varn])
            globals()[var] = dist

    # print(d_mae)
    d_mae = np.concatenate((np.reshape(globals()[('maesquare' + '-' + str(Dbase[0]))], (1, com_num)).T), axis=0)
    col = 0
    for din in range(len(Dbase)):
        dId = Dbase[din]
        base = str(dId)
        for run in range(len(maename)):
            var = 'mae' + maename[run] + '-' + str(base)
            if col > 0:
                d_mae = np.concatenate((d_mae.reshape(col, com_num), np.array(globals()[var]).reshape(1, com_num)),
                                       axis=0)
            col = col + 1
    print(d_mae.shape)
    df_mae = pd.DataFrame(d_mae, columns=stocks_yahoofinance_edited.index[:com_num])
    df_mae.index = indexname
    col = 0
    d_mse = np.concatenate((np.reshape(globals()[('msesquare' + '-' + str(Dbase[0]))], (1, com_num)).T), axis=0)
    for din in range(len(Dbase)):
        dId = Dbase[din]
        base = str(dId)
        for run in range(len(maename)):
            var = 'mse' + maename[run] + '-' + str(base)
            if col > 0:
                d_mse = np.concatenate((d_mse.reshape(col, com_num), np.array(globals()[var]).reshape(1, com_num)),
                                       axis=0)
            col = col + 1
    print(d_mse.shape)
    df_mse = pd.DataFrame(d_mse, columns=stocks_yahoofinance_edited.index[:com_num])
    df_mse.index = indexname

    d_dist = np.concatenate((globals()[('distsquare' + '-' + str(Dbase[0]))].reshape(1, com_num)), axis=0)
    col = 0
    for din in range(len(Dbase)):
        dId = Dbase[din]
        base = str(dId)
        for run in range(len(maename)):
            var = 'dist' + maename[run] + '-' + str(base)
            if col > 0:
                d_dist = np.concatenate((d_dist.reshape(col, com_num), globals()[var].reshape(1, com_num)), axis=0)
            col = col + 1
    print(d_dist.shape)
    df_dist = pd.DataFrame(d_dist, columns=stocks_yahoofinance_edited.index[:com_num])
    df_dist.index = indexname

    # pd.set_option('display.max_columns', None)
    # df_dist.to_csv(base+'dist.csv')
    # df_mae.to_csv(base+'mae.csv')
    # df_mse.to_csv(base+'mse.csv')
    # gap_df_mae = basecompare(df_mae)
    # print(gap_df_mae)
    # print(df_mae)
    # print(df_mse)

    return df_dist, df_mae, df_mse

def get_results_D_thencomp(df_dist, df_mae, df_mse):
    resindex = []
    for din in range(len(Dbase)):
        dId = Dbase[din]
        base = str(dId)
        resindex.append((str(1) + '-' + str(base)))

    gap_df_dist = basecompare(df_dist)
    gap_df_mae = basecompare(df_mae)
    gap_df_mse = basecompare(df_mse)
    gap = gap_df_dist * 0.1 + gap_df_mae
    g_df = gap_df_dist.loc[gap_df_dist.index.isin(resindex)]
    g_mae = gap_df_mae.loc[gap_df_mae.index.isin(resindex)]
    g_mse = gap_df_mse.loc[gap_df_mse.index.isin(resindex)]
    g_total = gap.loc[gap.index.isin(resindex)]
    return g_df, g_mae, g_mse, g_total



if __name__ == '__main__':
    df_dist, df_mae, df_mse = get_results_D()
    print(df_mse)  # output the dtw_distance results on different parameters
    # 0-WS (this is for the reference objective function-based results, and WS is the windowsize parameter)
    # 1-WS (this is for the proposed objective function-based results, and WS is the windowsize parameter)
    # get_results_D_thencomp(df_dist, df_mae, df_mse)--- this function is for remove the reference-based results,
#   it is easy to check the left framedata - which parameter is best for this stock.
