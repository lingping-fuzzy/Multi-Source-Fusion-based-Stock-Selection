from numpy import asarray
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# transform a time series dataset into a supervised learning dataset
from cost_function import squared_log_plus_one, load_data_yahoo


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(model, train, testX, valid=True, n_out=5):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-n_out], train[:, -n_out:]

    if valid == True:
        model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0], model


# Step 2 :
def predict_weekMonthly_byXGB_mixed(Symbols_, stocks_finance, storcks_edits, _nestimators=None, _nin=None, _nout=None,
                                    obj='reg:squarederror', eval=None, n_tests=None,
                                    test_size_monthly=None, test_size_weekly=None, Tdnum=None, evalf=None):
    ptr = 0
    weekly = [0] * storcks_edits.shape[0]
    monthly = [0] * storcks_edits.shape[0]
    MAEall = [0] * storcks_edits.shape[0]
    preds = {}
    trueYs = {}

    for item in Symbols_:
        # Step 2.0 :  fix random seed for reproducibility
        np.random.seed(7)
        print('here is predict company ', item)
        # Step 2.1 : load data (adjusted closed prices) & normalize
        data = stocks_finance[item].values
        # data = data[-100:]
        scaledata = MinMaxScaler(feature_range=(0, 1))
        data = data.astype('float32')
        data = data.reshape(-1, 1)
        data = scaledata.fit_transform(data)
        values = series_to_supervised(data, n_in=_nin, n_out=_nout)

        mae, y, pred, MonthlyME, WeeklyME = walk_forward_validation_predict(values, n_tests, \
                                                                            nestimators=_nestimators, \
                                                                            obj=obj, eval=eval, \
                                                                            test_size_monthly=test_size_monthly, \
                                                                            test_size_weekly=test_size_weekly, \
                                                                            n_out=_nout, Tdnum=Tdnum, \
                                                                            scaledata=scaledata, evalf=evalf)
        print('MAE: %.3f' % mae)
        preds[item] = pred
        trueYs[item] = y
        MAEall[ptr] = mae
        weekly[ptr] = WeeklyME
        monthly[ptr] = MonthlyME
        ptr += 1
    return weekly, monthly, MAEall, preds, trueYs


# walk-forward validation for univariate data
def walk_forward_validation_predict(data, n_test, nestimators=400, obj='reg:squarederror',
                                    eval=None, test_size_monthly=27, test_size_weekly=9,
                                    Tdnum=100, n_out=5, scaledata=None, evalf=None):
    train, test = train_test_split(data, n_test + Tdnum)  # the last 50 instance only for model test, no use for train
    model = XGBRegressor(objective=obj, n_estimators=nestimators, random_state=42, \
                         max_leaves=5, verbosity=0)

    history = [x for x in train]

    for i in range(len(test) - n_test):  # for testing only use the (n-100)->(n-50), and (n-50)->n use only test
        # split test row into input and output columns
        testX, testy = test[i, :-n_out], test[i, -n_out]
        # fit model on history and make a prediction
        yhat, model = xgboost_forecast(model, history, testX, valid=True, n_out=n_out)
        history.append(test[i])

    # estimate prediction error
    predictions = list()
    # split dataset

    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat, model = xgboost_forecast(model, history, testX, False)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # summarize progress

    predictions = np.array(predictions)
    # test = np.array(test)
    predictions = predictions.reshape(-1, 1)

    testT = test[:, -1].reshape(-1, 1)
    testPredict = scaledata.inverse_transform(predictions)
    testY = scaledata.inverse_transform(testT)
    error = mean_absolute_error(testY, testPredict)

    MonthlyME = np.mean(testPredict[-test_size_monthly:] - testY[-test_size_monthly:])
    WeeklyME = np.mean(testPredict[-test_size_weekly:] - testY[-test_size_weekly:])
    return error, testY, testPredict, MonthlyME, WeeklyME


def main_run_data(obj=None, n_in=27, evalf=None):
    import time
    start = time.time()
    print("The time used to execute this is given below")
    stock_alphafinance, stocks_alphafinance_edited = load_data_yahoo()

    weekly_xgb0, monthly_xgb0, MAEall_xgb0, preds_xgb0, trueYs_xgb0 = predict_weekMonthly_byXGB_mixed(
        stocks_alphafinance_edited.index, stock_alphafinance, stocks_alphafinance_edited,
        _nestimators=50, _nin=n_in, _nout=1, obj=obj, n_tests=50, test_size_monthly=27, test_size_weekly=9,
        Tdnum=100, evalf=evalf)  #
    return weekly_xgb0, monthly_xgb0, MAEall_xgb0, preds_xgb0, trueYs_xgb0


if __name__ == '__main__':
    inD = 5  # this is the window size parameter for dtw
    # obj = squared_log  # this is the cost function-objective function
    obj = squared_log_plus_one # this is the cost function-objective function (dtw+SLE)

    weekly_xgb0, monthly_xgb0, MAEall_xgb0, preds_xgb0, trueYs_xgb0 = \
        main_run_data(obj=obj, n_in=inD, evalf=None)
