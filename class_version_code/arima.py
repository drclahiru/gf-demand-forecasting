from class_version_code.forecast import Forecast
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


class Arima(Forecast):
    """
    A class for the ARIMA forecasting model. It consist of an auto-regressive model,
    a moving-average model and of a differencing component.

    Attributes
    ----------
    window_size : int
        the number of data-points used for training
    forecast_size : int
        the number of data-points that will be predicted after the last training data-point
    auto_regression : int
        the order to which we are executing the auto-regressive part of the model
    differencing : int
        the order to which we are differencing the data
    moving_average : int
        the order to which we are executing the moving average part of the model
    do_print : bool
        a flag that decides if we will print the method summary
    """

    def __init__(self, window_size, forecast_size, auto_regression, differencing, moving_average, do_print=False):
        Forecast.__init__(self, window_size, forecast_size, "ARIMA Model")
        self.auto_regression_order = auto_regression
        self.differencing_order = differencing
        self.moving_average_order = moving_average
        self.do_print = do_print
        self.predictions = [0.0]*forecast_size

    def forecast(self, unit_data):
        """
        Forecasts the time series for the given number of data-points.

        Parameters
        ----------
        unit_data : Series
            the entire time series to be forecast
        """
        # separate the time series into training and testing data
        self.train = unit_data.iloc[:self.window_size]
        self.train = self.train.asfreq('MS')
        self.train = self.train.astype('int')
        self.test = unit_data.iloc[self.window_size - 1:self.window_size + self.forecast_size - 1]
        self.test.index = pd.to_datetime(self.test.index)
        # initalize the model
        model = ARIMA(self.train.values, order=(self.auto_regression_order, self.differencing_order, self.moving_average_order))
        # fit the model
        model_fit = model.fit()
        # make predictions
        self.predictions = model_fit.forecast(self.forecast_size)
        # print model summary
        if self.do_print:
            print(model_fit.summary())
