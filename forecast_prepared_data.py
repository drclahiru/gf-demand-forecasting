import pandas as pd
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

SOURCE_PATH = 'prepared_data\\ALPHS_10Y_Prepared.csv'
MAT_GROUP = "ALPHS"
MAT_COL_NAME = "MD Material Group"

# holt or arima or lstm or all
MODEL = "lstm"

WINDOW_SIZE = 85
FORECAST_SIZE = 3
SMOOTH_LVL = .7
SMOOTH_SLOPE = 0.001
DAMPED_TREND = 0.98

AUTO_REGRESSION = 1
DIFFERENCING = 1
MOVING_AVREAGE = 1

NUM_OF_EPOCH = 50
NEURON_SIZE = 8

PLOT_SIZE = 15


def plot_data(train, test, predictions, unit_max):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, color="gray")
    ax.plot(train.index, train.values, color='gray')
    ax.plot(test.index, predictions[0], label="holts method", color='blue')
    ax.plot(test.index, predictions[1], label="ARIMA model", color='orange')
    ax.plot(test.index, predictions[2], label="LSTM model", color='green')
    ax.vlines(train.index[-1], 0, unit_max, linestyle='--', color='r',
              label='Start of forecast')
    plt.legend()


def plot_one_method(train, test, predictions, unit_max, method_label):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, color="gray")
    ax.plot(train.index, train.values, color='gray')
    ax.plot(test.index, predictions, label=method_label, color='blue')
    ax.vlines(train.index[-1], 0, unit_max, linestyle='--', color='r',
              label='Start of forecast')
    plt.legend()


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()
    return model


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# invert differenced value
def inverse_difference(history, y_hat, interval=1):
    return y_hat + history[-interval]


# create a differenced series
def difference(dataset, interval=1):
    # Fickey Fuller Test
    print("Summary Statistics - ADF Test For Stationarity\n")
    if stationarity_test(X=dataset, return_p=True, print_res=False) > 0.05:
        print("P Value is high. Differencing needed: " + str(
            stationarity_test(X=dataset, return_p=True, print_res=False)))
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)
    else:
        stationarity_test(X=dataset)
        return dataset


def stationarity_test(X, log_x="Y", return_p=False, print_res=True):
    if log_x == "Y":
        X = np.log(X[X > 0].astype(np.float32))
    dickey_fuller = adfuller(X)
    if print_res:
        # If ADF statistic is < our 1% critical value (sig level) we can conclude it's not a fluke (ie low P val /
        # reject H(0))
        print('ADF Stat is: {}.'.format(dickey_fuller[0]))
        # A lower p val means we can reject the H(0) that our data is NOT stationary
        print('P Val is: {}.'.format(dickey_fuller[1]))
        print('Critical Values (Significance Levels): ')
        for key, val in dickey_fuller[4].items():
            print(key, ":", round(val, 3))
    if return_p:
        return dickey_fuller[1]


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def lstm_forecasting_3_mnt(unit_data):
    raw_unit_data = unit_data.values
    diff_unit_data = difference(raw_unit_data, 1)
    supervised_unit_data = timeseries_to_supervised(diff_unit_data, 1)
    sup_unit_data_val = supervised_unit_data.values
    train, test = sup_unit_data_val[0:WINDOW_SIZE - 1], sup_unit_data_val[WINDOW_SIZE - 1:]
    scaler, train_scaled, test_scaled = scale(train, test)
    lstm_model = fit_lstm(train_scaled, 1, NUM_OF_EPOCH, NEURON_SIZE)
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        X = X.reshape(1, 1, len(X))
        y_hat = lstm_model.predict(X, batch_size=1)
        # invert scaling
        y_hat = invert_scale(scaler, X, y_hat)
        # invert differencing
        y_hat = inverse_difference(raw_unit_data, y_hat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(y_hat)
    return predictions[:FORECAST_SIZE], "LSTM model"


def arima_forecasting_3_mnt(unit_data):
    train = unit_data.iloc[:WINDOW_SIZE]
    train = train.asfreq('MS')
    train = train.astype('int')
    model = ARIMA(train.values, order=(AUTO_REGRESSION, DIFFERENCING, MOVING_AVREAGE))
    model_fit = model.fit()
    predictions = model_fit.forecast(3)
    print(model_fit.summary())
    return predictions, "ARIMA model"


def holts_dampening_forecasting_3_mnt(unit_data):
    train = unit_data.iloc[:WINDOW_SIZE]
    train = train.asfreq('MS')
    model = Holt(train.values, damped_trend=True)
    model._index = pd.to_datetime(train.index)
    model_fit = model.fit(smoothing_level=SMOOTH_LVL, smoothing_trend=SMOOTH_SLOPE, damping_trend=DAMPED_TREND)
    predictions = model_fit.forecast(FORECAST_SIZE)
    print(model_fit.summary())
    return predictions, "Holts method"


def calculate_relative_error(predictions, test):
    forecast_errors = [abs(test[i] - predictions[i]) / max(test[i], predictions[i]) for i in range(len(test))]
    bias = sum(forecast_errors) * 1.0 / len(test)
    return bias


def main():
    # initialization
    data = pd.read_csv(SOURCE_PATH, header=0)
    filtered_data = data[data[MAT_COL_NAME] == MAT_GROUP].values[0][1:]
    new_index = pd.to_datetime(data.columns[1:])
    unit_data = pd.Series(filtered_data, new_index)
    if MODEL == 'arima':
        predictions, label = arima_forecasting_3_mnt(unit_data)
    elif MODEL == 'holt':
        predictions, label = holts_dampening_forecasting_3_mnt(unit_data)
    elif MODEL == 'lstm':
        predictions, label = lstm_forecasting_3_mnt(unit_data)
    else:
        lstm_predictions = lstm_forecasting_3_mnt(unit_data)
        arima_predictions = arima_forecasting_3_mnt(unit_data)
        holts_predictions = holts_dampening_forecasting_3_mnt(unit_data)
    train = unit_data.iloc[WINDOW_SIZE - PLOT_SIZE:WINDOW_SIZE]
    train = train.asfreq('MS')
    test = unit_data.iloc[WINDOW_SIZE - 1:WINDOW_SIZE + FORECAST_SIZE - 1]
    test.index = pd.to_datetime(test.index)
    print(f"Real values: \n{test.values}")
    if MODEL == 'all':
        print()
        print(f"Holts Predictions: \n{holts_predictions}")
        print(f"ARIMA Predictions: \n{arima_predictions}")
        print(f"LSTM Predictions: \n{lstm_predictions}")
        lstm_rel_error = calculate_relative_error(lstm_predictions, test.values)
        arima_rel_error = calculate_relative_error(arima_predictions, test.values)
        holt_rel_error = calculate_relative_error(holts_predictions, test.values)
        print()
        print(f"holts Relative Error: {holt_rel_error}")
        print(f"ARIMA Relative Error: {arima_rel_error}")
        print(f"LSTM Relative Error: {lstm_rel_error}")
        plot_data(train, test, [holts_predictions, arima_predictions, lstm_predictions], max(filtered_data))
        plt.show()
    else:
        print()
        print(f"Predictions: \n{predictions}")
        rel_error = calculate_relative_error(predictions, test.values)
        print()
        print(f"Relative Error: {rel_error}")
        plot_one_method(train, test, predictions, max(filtered_data), label)
        plt.show()


if __name__ == "__main__":
    main()
