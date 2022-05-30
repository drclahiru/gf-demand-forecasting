import math

import pandas as pd
from datetime import datetime

class Forecast:
    """
    The main class for forecasting methods/models

    Attributes
    ----------
    name : str
        the name of the specific forecasting method/model that uses this main class
    window_size : int
        the number of data-points used for training
    forecast_size : int
        the number of data-points that will be predicted after the last training data-point
    predictions : list
        a list of predictions that are the result of the trained forecasting method/model
    train : Series
        a data series that was used to train the forecasting method/model
    test : Series
        a data series that was used to test the result of the forecasting method/model
    rel_error : float
        a number that represents the relative error between the predictions and the test data
    """

    def __init__(self, window_size, forecast_size, name):
        self.name = name
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.predictions = [0.0] * forecast_size
        self.train = None
        self.test = None
        self.rel_error = None

    def calculate_relative_error(self):
        """
        Calculates the relative error between the forecasted predictions and
        the time series data-points used for testing

        """
        forecast_errors = [abs(abs(self.test[i]) - abs(self.predictions[i])) / max(abs(self.test[i]), abs(self.predictions[i]))
                           for i in range(len(self.test))]
        rel_error = sum(forecast_errors) * 1.0 / len(self.test)
        self.rel_error = rel_error

    def calculate_mase(self):
        forecast_errors = [abs(self.test[i] - self.predictions[i]) for i in range(len(self.test))]
        mean_absolute_error = sum(forecast_errors) / len(self.test)
        naive_forecast_errors = [abs(self.test[i] - self.test[i-1]) for i in range(1, len(self.test))]
        mean_absolute_error_naive = sum(naive_forecast_errors) / (len(self.test)-1)
        mas_error = mean_absolute_error / mean_absolute_error_naive
        self.rel_error = mas_error

    def calculate_rmse(self):
        forecast_errors = [(self.test[i] - self.predictions[i])**2 for i in range(len(self.test))]
        ms_error = sum(forecast_errors) / len(self.test)
        rms_error = math.sqrt(ms_error)
        self.rel_error = rms_error

    def divide_data(self, unit_data):
        """
        Separates the time series data-points into series for training and for testing

        Parameters
        ----------
        unit_data : Series
            the entire time series to be seperated
        """
        self.train = unit_data.iloc[:self.window_size]
        self.train = self.train.asfreq('MS')
        self.test = unit_data.iloc[self.window_size - 1:self.window_size + self.forecast_size - 1]
        self.test.index = pd.to_datetime(self.test.index)

    def print_result(self):
        """
        The printing method for easy overview of the forecasting

        """
        # print forecasting method used
        print(f"\n\nAPPROACH NAME:\n\t{self.name}\n")
        # print the real data and the predictions
        print("DATE\t\tReal Values\t\tPredictions")
        for i in range(len(self.predictions)):
            date = self.test.index[i].strftime("%Y-%m")
            print(f"{date}\t\t{round(self.test.values[i])}\t\t\t{round(self.predictions[i])}")
        # print the relative error
        print(f"\nRelative Error:\n\t{self.rel_error}\n")
        print("===========================================")
