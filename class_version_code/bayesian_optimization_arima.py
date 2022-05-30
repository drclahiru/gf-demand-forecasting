import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from class_version_code.arima import Arima
from class_version_code.exec_script_grid_search_arima import combine_material_groups
from class_version_code.holt import Holts
from class_version_code.plot_data import PlotData

import time
from datetime import timedelta
start_time = time.monotonic()

# ignoring warnings from statsmodels and skopt
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)




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
MODEL = "arima"

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
dim_auto_regression = Integer(low=1, high=20, name='auto_regression')
dim_differencing = Integer(low=1, high=20, name='differencing')
dim_moving_average = Integer(low=1, high=20, name='moving_average')

dimensions = [dim_auto_regression, dim_differencing, dim_moving_average]

default_parameters = [1, 1, 1]
best_rel_error = 1.0


@use_named_args(dimensions=dimensions)
def fitness(auto_regression, differencing, moving_average):
    """
    initialize the Arima model and calculate the relative error

    Parameters
    ----------
    auto_regression : int
        the order to which we are executing the auto-regressive part of the model
    differencing : int
        the order to which we are differencing the data
    moving_average : int
        the order to which we are executing the moving average part of the model

    Returns
    ----------
    rel_error: float
        the relative error for the given set of hyper parameters
    """
    # Print the hyper parameters
    # print('auto_regression:', auto_regression)
    # print('differencing:', differencing)
    # print('moving_average:', moving_average)
    # print()

    # initialize the forecasting model
    arima_model = Arima(TRAINING_SIZE, FORECAST_SIZE, auto_regression, differencing, moving_average, do_print=DO_PRINT)
    data = pd.read_csv(SOURCE_PATH, header=0)
    unit_data = combine_material_groups(data)
    arima_model.forecast(unit_data)
    arima_model.calculate_relative_error()
    rel_error = arima_model.rel_error

    # print("Relative error:" + str(rel_error) + "\n")

    global best_rel_error
    global best_auto_regression
    global best_differencing
    global best_moving_average

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_auto_regression = auto_regression
        best_differencing = differencing
        best_moving_average = moving_average

        # predictions = [0.0, grundfos_forecasting(data, arima_model, FORECAST_SIZE), 0.0]
        # # divide the data to be used for plotting
        # arima_model.divide_data(unit_data)
        # # initialize the data plot class
        # plot_data = PlotData(PLOT_SIZE, arima_model.train, arima_model.test, predictions)

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
    print("auto_regression: ", best_auto_regression)
    print("differencing: ", best_differencing)
    print("moving_average: ", best_moving_average)

    end_time = time.monotonic()
    print("Duration: ", timedelta(seconds=end_time - start_time))

    print("**************************")
    print("Product group: ", SOURCE_PATH[21:-17])
    print(best_auto_regression,";", best_differencing,";", best_moving_average)


if __name__ == "__main__":
    main()
