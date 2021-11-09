from class_version_code.forecast import Forecast
from statsmodels.tsa.api import Holt
import pandas as pd


class Holts(Forecast):
    """
    A class for the Holts forecasting method. This is an exponential smoothing class and it incorporates
    Simple Exponential Smoothing as well as Exponential Smoothing with trend and possible dampening of the trend.

    Attributes
    ----------
    window_size : int
        the number of data-points used for training
    forecast_size : int
        the number of data-points that will be predicted after the last training data-point
    smooth_lvl : float
        a parameter that determines how much of old data will be used for forecasting (alpha)
    smooth_slope : float
        a parameter that determines how much will the forecast follow the trend (beta)
    damped_trend : float
        a parameter that determines how much will the forecasted trend be dampened (gamma)
    is_damped : bool
        a flag that determines if the forecasted trend will be dampened
    do_print : bool
        a flag that decides if we will print the method summary
    """

    def __init__(self, window_size, forecast_size, smooth_lvl, smooth_slope, damped_trend, is_damped=True,
                 do_print=False):
        Forecast.__init__(self, window_size, forecast_size, "Holts Method")
        self.smooth_lvl = smooth_lvl
        self.smooth_slope = smooth_slope
        self.damped_trend = damped_trend
        self.is_damped = is_damped
        self.do_print = do_print

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
        self.test = unit_data.iloc[self.window_size - 1:self.window_size + self.forecast_size - 1]
        self.test.index = pd.to_datetime(self.test.index)
        # initalize the model
        model = Holt(self.train.values, damped_trend=self.is_damped)
        model._index = pd.to_datetime(self.train.index)
        # fit the model
        if self.is_damped:
            model_fit = model.fit(smoothing_level=self.smooth_lvl, smoothing_trend=self.smooth_slope,
                                  damping_trend=self.damped_trend)
        else:
            model_fit = model.fit(smoothing_level=self.smooth_lvl, smoothing_trend=self.smooth_slope)
        # make predictions
        self.predictions = model_fit.forecast(self.forecast_size)
        # print model summary
        if self.do_print:
            print(model_fit.summary())
