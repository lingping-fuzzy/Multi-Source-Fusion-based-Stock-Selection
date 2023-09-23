import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")  # don't spam the notebook with warnings


def load_data_yahoo():
    stock_alphafinance = pd.read_csv('data/stocks_yahoofinance.csv')
    stock_alphafinance = stock_alphafinance.set_index('Date')

    stocks_alphafinance_edited = pd.read_csv('data/stocks_yahoofinance_edited.csv')
    stocks_alphafinance_edited.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
    stocks_alphafinance_edited = stocks_alphafinance_edited.set_index('Symbol')
    print(stocks_alphafinance_edited.index)
    print(stocks_alphafinance_edited.shape)
    return stock_alphafinance, stocks_alphafinance_edited


# https://github.com/dmlc/xgboost/blob/54582f641ad102ffd09944b7c9c93f1fa055fe0c/doc/tutorials/custom_metric_obj.rst
def gradient_sl(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    # y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(dtrain)) / (predt + 1)


def hessian_sl(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    # y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(dtrain) + 1) /
            np.power(predt + 1, 2))


def squared_log(dtrain: np.ndarray,
                predt: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient_sl(predt, dtrain)
    hess = hessian_sl(predt, dtrain)
    return grad, hess


def dtw_distance_predmulti(data, preds_):
    # from main import used_len  #we set 100
    from dtw_nofuture import dtw_measure
    # from main import used_D  #(n_out) we set 27
    n_out = 27
    dist = np.zeros(len(data))
    # startid = len(data) - used_len()
    startid = len(data) - 100
    temp = int(n_out / 2)
    for i in range(startid, len(data) - n_out, temp):
        s1 = np.array(preds_[i:(i + n_out)])
        s2 = np.array(data[i:(i + n_out)])

        DD, PATH, distance = dtw_measure(s1, s2, w=5)
        dist[i + n_out: i + n_out + temp] = distance / n_out  # *0.1
    return dist


def squared_log_plus_one(dtrain: np.ndarray,
                         predt: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient_sl(predt, dtrain)
    hess = hessian_sl(predt, dtrain)
    dist = dtw_distance_predmulti(dtrain, predt)
    return grad + dist * 0.1, hess + dist * 0.1
