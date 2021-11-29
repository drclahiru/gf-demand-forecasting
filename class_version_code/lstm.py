from class_version_code.forecast import Forecast
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator


def build_exp_decay(epochs):
    decay = 0.5 ** (2 / epochs)

    def exp_decay(epoch):
        lr = 0.001 * decay ** epoch
        return lr

    return exp_decay


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
    look_back : int
        the number of months that will go as input to the lstm
    do_print : bool
        a flag that decides if we will print the method summary
    """

    def __init__(self, window_size, forecast_size, num_of_epoch, num_of_neurons, look_back, do_print=False):
        Forecast.__init__(self, window_size, forecast_size, "LSTM Model")
        self.num_of_epoch = num_of_epoch
        self.num_of_neurons = num_of_neurons
        self.look_back = look_back
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
        # scale the training data into a range from -1 to 1
        train_data = self.train.values
        train_data = train_data.reshape((-1, 1))
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        # Take a sequence of data-points gathered at equal intervals,
        # to turn the sequence into a supervised learning problem
        train_generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=self.look_back,
                                              batch_size=1)
        # defining the model
        model = Sequential()
        model.add(
            LSTM(self.num_of_neurons,
                 activation='relu',
                 input_shape=(self.look_back, 1))
        )
        model.add(Dense(1))
        # add learning rate decay to stabilize the training process
        l_rate = ExponentialDecay(0.001, decay_steps=len(train_generator), decay_rate=0.5 ** (2.0 / self.num_of_epoch))
        adam = Adam(learning_rate=l_rate)
        model.compile(optimizer=adam, loss='mse')
        # fitting the model with the generator and deciding whether we want to print the progress
        model.fit_generator(train_generator, epochs=self.num_of_epoch, verbose=(1 if self.do_print else 0))
        # forecast
        prediction_list = scaled_train_data[-self.look_back:]
        for _ in range(self.forecast_size - 1):
            x = prediction_list[-self.look_back:]
            x = x.reshape((1, self.look_back, 1))
            # predict for the next month
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[self.look_back - 1:]
        prediction_list = scaler.inverse_transform(prediction_list.reshape((-1, 1)))
        prediction_list = prediction_list.flatten()

        self.predictions = prediction_list
