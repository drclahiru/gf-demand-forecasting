from class_version_code.forecast import Forecast
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator


def inverse_difference(history, y_hat, interval=1):
    """
    invert differenced value

    Parameters
    ----------
    history : array
        the values of the entire time series
    y_hat : float
        the prediction of the forecasting
    interval: bool
        the interval from the end of the data that is used to find the data-point to be added to the prediction

    Returns
    ----------
    y_hat + history[-interval] : float
        the inverse differenced predicted value
    """
    return y_hat + history[-interval]


def stationarity_test(X, log_x="Y", return_p=False, print_res=True):
    """
    test whether the data is stationary or not

    Parameters
    ----------
    X : array
        the time series values to be tested for stationarity
    log_x : str
        string to decide if we need to log the data and how
    return_p: bool
        a flag that tells us if we ant to return the P value of the test
    print_res: bool
        a flag that determines if we want to print the result of the test

    Returns
    ----------
    dickey_fuller[1] : float
        the P value of the test
    """
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


def timeseries_to_supervised(data, lag=1):
    """
    frame a sequence as a supervised learning problem

    Parameters
    ----------
    data : array
        the time series values to be turned into a supervised learning problem
    lag : str
        the lag for which the data is shifted

    Returns
    ----------
    df : array
        the data now prepared to be used for a supervised learning problem
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


class Lstm(Forecast):
    """
    A class for the LSTM forecasting model. It is a recurrent neural network.

    Attributes
    ----------
    window_size : int
        the number of data-points used for training
    forecast_size : int
        the number of data-points that will be predicted after the last training data-point
    num_of_epoch : int
        the number of epochs used to train the neural network
    num_of_neurons : int
        the number of neurons in the neural network architecture
    do_print : bool
        a flag that decides if we will print the method summary
    """

    def __init__(self, window_size, forecast_size, num_of_epoch, num_of_neurons, do_print=False):
        Forecast.__init__(self, window_size, forecast_size, "LSTM Model")
        self.num_of_epoch = num_of_epoch
        self.num_of_neurons = num_of_neurons
        self.do_print = do_print
        self.scaler = None

    def forecast(self, unit_data):
        """
        Forecasts the time series for the given number of data-points.

        Parameters
        ----------
        unit_data : Series
            the entire time series to be forecast
        """
        # separate the time series into training and testing parts
        self.train = unit_data.iloc[:self.window_size]
        self.train = self.train.asfreq('MS')
        self.test = unit_data.iloc[self.window_size - 1:self.window_size + self.forecast_size - 1]
        self.test.index = pd.to_datetime(self.test.index)
        # prepare the data for LSTM training
        raw_unit_data = unit_data.values
        # possibly use differencing on data
        diff_unit_data = self.difference(raw_unit_data, 1)
        # turn this into a supervised learning problem
        supervised_unit_data = timeseries_to_supervised(diff_unit_data, 1)
        sup_unit_data_val = supervised_unit_data.values
        train, test = sup_unit_data_val[0:self.window_size - 1], sup_unit_data_val[self.window_size - 1:]
        # scale the data
        train_scaled, test_scaled = self.scale(train, test)
        # fit the model
        lstm_model = self.fit_lstm(train_scaled, 1)
        train_reshaped = train_scaled[:, 0].reshape(1, len(train_scaled), 1)
        # make the predictions
        lstm_model.predict(train_reshaped, batch_size=1)
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            X = X.reshape(1, 1, len(X))
            y_hat = lstm_model.predict(X, batch_size=1)
            # invert scaling
            y_hat = self.invert_scale(X, y_hat)
            # invert differencing
            y_hat = inverse_difference(raw_unit_data, y_hat, len(test_scaled) + 1 - i)
            # store forecast
            predictions.append(y_hat)
        self.predictions = predictions[:self.forecast_size]

    def fit_lstm(self, train, batch_size):
        """
        Creates the LSTM architecture and fits it.

        Parameters
        ----------
        train : Series
            the prepared data series to use ofr training
        batch_size : int
            batch size used for fitting the model

        Returns
        ----------
        model : Sequential
            the fitted model
        """
        # prepare the data
        close_data = self.train.values.reshape((-1, 1))
        close_data = np.asarray(close_data).astype('float32')
        train_generator = TimeseriesGenerator(close_data, close_data, length=self.window_size-1, batch_size=20)
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(1, X.shape[0], X.shape[1])
        # build the model
        model = Sequential()
        model.add(LSTM(self.num_of_neurons, input_shape=(self.window_size-1, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit the model
        for i in range(self.num_of_epoch):
            # if we want to print the progress of fitting
            if self.do_print:
                model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False, verbose=1)
            else:
                model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)
            model.reset_states()
        return model

    def invert_scale(self, value):
        """
        inverse scaling for a forecasted value

        Parameters
        ----------
        X : Series
            data to be inverse scaled
        value : int
            the value for which the data is being inverse scaled

        Returns
        ----------
        inverted[0, -1] : array
            the inverse scaled data
        """
        new_row = [value] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        return inverted[0, -1]

    def scale(self, train, test):
        """
        scale train and test data to [-1, 1]

        Parameters
        ----------
        train : array
            the training data to be scaled
        test : array
            the testing data to be scaled

        Returns
        ----------
        train_scaled : array
            the scaled training data
        test_scaled : array
            the scaled testing data
        """
        # fit scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = self.scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = self.scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = self.scaler.transform(test)
        return train_scaled, test_scaled

    def difference(self, dataset, interval=1):
        """
        create a differenced series

        Parameters
        ----------
        dataset : array
            the values of the entire time series
        interval : int
            the interval at which the data is being differenced

        Returns
        ----------
        dataset : array
            the differenced dataset
        """
        # Fickey Fuller Test to see if we need to do differencing
        if self.do_print:
            print("Summary Statistics - ADF Test For Stationarity\n")
        # everything with p value above 0.05 needs to be differenced
        if stationarity_test(X=dataset, return_p=True, print_res=False) > 0.05:
            if self.do_print:
                print("P Value is high. Differencing needed: " + str(
                    stationarity_test(X=dataset, return_p=True, print_res=False)))
            # the differencing process
            diff = list()
            for i in range(interval, len(dataset)):
                # subtracting wto data-points that are for the given interval apart
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return pd.Series(diff)
        stationarity_test(X=dataset, print_res=self.do_print)
        return dataset