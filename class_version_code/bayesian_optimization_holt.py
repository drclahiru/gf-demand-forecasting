import matplotlib.pyplot as plt
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from statsmodels.tsa import arima_model

from class_version_code.exec_script_grid_search_holts import combine_material_groups
from class_version_code.holt import Holts
from class_version_code.plot_data import PlotData

import time
from datetime import timedelta
start_time = time.monotonic()



# Choose whether you want to forecast for the whole product group or just for material group
# Choices:
#   "PG Global" - for forecasting on sum of material groups
#   "PG" - for forecasting on all material groups and then summing
#   material group name (e.g. "ALPHS") - for forecasting on some material group
#   "Grundfos" - for Grundfos forecasting
MAT_GROUP = "Grundfos"

# Name of the column in which the material group names are
MAT_COL_NAME = "MD Material Group"

# Choose method or model to use for forecasting
# Choices:
#   "holt" - for Holts Method (Exponential Smoothing)
#   "arima" - for ARIMA Model (auto regressive model)
#   "lstm" - for LSTM Model (Recurrent Neural Network)
#   "all" - for all of the methods and models
MODEL = "holt"

# path from which we extract product group data
SOURCE_PATH = '../prepared_data/DBS_CIRSC_10Y_Prepared.csv'

# parameters for forecasting
TRAINING_SIZE = 102
FORECAST_SIZE = 9

# parameters for Holts method
SMOOTH_LVL = .6
SMOOTH_SLOPE = .5
DAMPED_TREND = .2

# parameters for the ARIMA model
AUTO_REGRESSION = 1
DIFFERENCING = 1
MOVING_AVREAGE = 3

# parameters for the LSTM model
NUM_OF_EPOCH = 100
OUTPUT_SIZE = 8
LOOK_BACK = 10

# how much data-points to show on plot before forecasting
PLOT_SIZE = 15

# do we print the method and model summaries as well as their progression
DO_PRINT = False

# hyper parameters as ranges
dim_smooth_lvl = Real(low=0.000001, high=1.0, name='smooth_lvl')
dim_smooth_slope = Real(low=0.000001, high=1.0, name='smooth_slope')
dim_damped_trend = Real(low=0.000001, high=1.0, name='damped_trend')

dimensions = [dim_smooth_lvl, dim_smooth_slope, dim_damped_trend]

default_parameters = [0.000001, 0.000001, 0.000001]
best_rel_error = 1.0


@use_named_args(dimensions=dimensions)
def fitness(smooth_lvl, smooth_slope, damped_trend):
    """
    initialize the Holt's method and calculate the relative error

    Parameters
    ----------
    smooth_lvl : float
        a parameter that determines how much of old data will be used for forecasting (alpha)
    smooth_slope : float
        a parameter that determines how much will the forecast follow the trend (beta)
    damped_trend : float
        a parameter that determines how much will the forecasted trend be dampened (gamma)

    Returns
    ----------
    rel_error: float
        the relative error for the given set of hyper parameters
    """
    # Print the hyper parameters
    # print('smooth_lvl:', smooth_lvl)
    # print('smooth_slope:', smooth_slope)
    # print('damped_trend:', damped_trend)
    # print()

    # initialize the forecasting method
    holts_model = Holts(TRAINING_SIZE, FORECAST_SIZE, smooth_lvl, smooth_slope, damped_trend, do_print=DO_PRINT)
    data = pd.read_csv(SOURCE_PATH, header=0)
    unit_data = combine_material_groups(data)
    holts_model.forecast(unit_data)
    holts_model.calculate_relative_error()
    rel_error = holts_model.rel_error

    # print("Relative error:" + str(rel_error) + "\n")

    global best_rel_error
    global best_smooth_lvl
    global best_smooth_slope
    global best_damped_trend

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_smooth_lvl = smooth_lvl
        best_smooth_slope = smooth_slope
        best_damped_trend = damped_trend

        # predictions = [grundfos_forecasting(data, holts_model, FORECAST_SIZE), 0.0, 0.0]
        # # divide the data to be used for plotting
        # holts_model.divide_data(unit_data)
        # # initialize the data plot class
        # plot_data = PlotData(PLOT_SIZE, holts_model.train, holts_model.test, predictions)
        #
        # # plot data with all of the forecasting method/model predictions
        # if MODEL == 'all':
        #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
        #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
        # # plot data for one forecasting method/model prediction
        # else:
        #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        # plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        # plt.show()

    return rel_error


def main():
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI',  # Expected Improvement.
                                n_calls=100, x0=default_parameters)

    print("**************************")
    print("Product group: ", SOURCE_PATH[21:-17])
    print("Forecast size: ", FORECAST_SIZE)
    # print the best relative error
    print("Best relative error:** ", best_rel_error)
    # print the best hyper parameters
    print("The best hyper parameters are: ")
    print("smooth_lvl: ", best_smooth_lvl)
    print("smooth_slope: ", best_smooth_slope)
    print("damped_trend: ", best_damped_trend)

    end_time = time.monotonic()
    print("Duration: ", timedelta(seconds=end_time - start_time))

    print("**************************")
    print("Product group: ", SOURCE_PATH[21:-17])
    print(round(best_smooth_lvl,1),";", round(best_smooth_slope,1),";", round(best_damped_trend,1))


if __name__ == "__main__":
    main()
