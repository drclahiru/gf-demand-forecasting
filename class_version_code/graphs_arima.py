import math
import warnings
import pandas as pd
import os
import numpy as np
import statistics as stat

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from tensorflow.keras import backend as K
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignoring warnings and info logs from tensorflow
from class_version_code.exec_script_grid_search_arima import combine_material_groups, forecast_data, total_print, \
    grundfos_forecasting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from class_version_code.arima import Arima
from class_version_code.holt import Holts
from class_version_code.lstm import Lstm
from class_version_code.plot_data import PlotData
import matplotlib.pyplot as plt
from tqdm import tqdm

# ignoring warnings from statsmodels
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

grid_search_predictions = []
bayesian_predictions = []

# path from which we extract product group data
SOURCE_PATH = '..\\prepared_data\\DBS_KP_10Y_Prepared.csv'

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


def run(auto_regression, differencing, moving_average, forecast_size):
    global grid_search_predictions
    # read the prepared data from the cvs file
    data = pd.read_csv(SOURCE_PATH, header=0)
    # initialize the forecasting methods/models
    arima_model = Arima(TRAINING_SIZE, forecast_size, auto_regression, differencing, moving_average, do_print=DO_PRINT)
    holts_model = Holts(TRAINING_SIZE, FORECAST_SIZE, SMOOTH_LVL, SMOOTH_SLOPE, DAMPED_TREND, is_damped=True,
                        do_print=DO_PRINT)
    lstm_model = Lstm(TRAINING_SIZE, FORECAST_SIZE, NUM_OF_EPOCH, OUTPUT_SIZE, LOOK_BACK, do_print=DO_PRINT)
    # forecast on the summed up product group data
    if MAT_GROUP == "PG Global":
        unit_data = combine_material_groups(data)
        predictions = forecast_data(holts_model, arima_model, lstm_model, unit_data)
        holts_model.divide_data(unit_data)
    # forecast on each material group and then add them
    elif MAT_GROUP == 'PG':
        print('PG')
        total_predictions = [[0.0] * FORECAST_SIZE, [0.0] * FORECAST_SIZE, [0.0] * FORECAST_SIZE]
        for material_group in tqdm(data[MAT_COL_NAME]):
            filtered_data = data[data[MAT_COL_NAME] == material_group].values[0][1:]
            new_index = pd.to_datetime(data.columns[1:])
            unit_data = pd.Series(filtered_data, new_index)
            predictions = forecast_data(holts_model, arima_model, lstm_model, unit_data, calc_error=False)
            for i in range(len(predictions)):
                total_predictions[i] = [x + y for x, y in zip(total_predictions[i], predictions[i])]
        predictions = total_predictions
        unit_data = combine_material_groups(data)
        total_print(holts_model, arima_model, lstm_model, predictions, unit_data)
    # forecast with the Grundfos approach
    elif MAT_GROUP == 'Grundfos':
        predictions = [0.0, grundfos_forecasting(data, arima_model, forecast_size), 0.0]
        unit_data = combine_material_groups(data)
        total_print(holts_model, arima_model, lstm_model, predictions, unit_data)
    # forecast for a single material group
    else:
        filtered_data = data[data[MAT_COL_NAME] == MAT_GROUP].values[0][1:]
        new_index = pd.to_datetime(data.columns[1:])
        unit_data = pd.Series(filtered_data, new_index)
        predictions = forecast_data(holts_model, arima_model, lstm_model, unit_data)
    # divide the data to be used for plotting
    holts_model.divide_data(unit_data)
    grid_search_predictions = predictions
    # initialize the the data plot class
    # plot_data = PlotData(PLOT_SIZE, holts_model.train, holts_model.test, predictions)
    # # plot data with all of the forecasting method/model predictions
    # if MODEL == 'all':
    #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
    #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
    # # plot data for one forecasting method/model prediction
    # else:
    #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
    # plt.show()
    return arima_model.rel_error


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
    global bayesian_predictions

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_auto_regression = auto_regression
        best_differencing = differencing
        best_moving_average = moving_average

        bayesian_predictions = grundfos_forecasting(data, arima_model, FORECAST_SIZE)
        predictions = [0.0, grid_search_predictions[1], bayesian_predictions]
        # predictions = [0.0, grundfos_forecasting(data, arima_model, FORECAST_SIZE), 0.0]
        # divide the data to be used for plotting
        arima_model.divide_data(unit_data)
        # initialize the data plot class
        plot_data = PlotData(PLOT_SIZE, arima_model.train, arima_model.test, predictions)

        # # plot data with all of the forecasting method/model predictions
        # if MODEL == 'all':
        #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
        #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
        # # plot data for one forecasting method/model prediction
        # else:
        #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        plot_data.plot_optimization(f"ARIMA {FORECAST_SIZE} months forecast for {SOURCE_PATH[21:-17]} product group")
        plt.show()

    return rel_error


def grid_search():
    # Grid Search for best parameters for a single horizon
    # choose the horizon by changing the FORECAST_SIZE variable
    best_rel_error = [2.0, 0.0, 0.0, 0.0]
    for auto_reg in tqdm(np.arange(0, 10, 1)):
        for diff in np.arange(0, 10, 1):
            for mov_avg in np.arange(0, 10, 1):
                temp_rel_error = run(auto_reg, diff, mov_avg, FORECAST_SIZE)
                if temp_rel_error < best_rel_error[0]:
                    best_rel_error = [temp_rel_error, auto_reg, diff, mov_avg]


def main():
    grid_search()
    gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=100, x0=default_parameters)


if __name__ == "__main__":
    # paths = ['..\\prepared_data\\DBS_SUPM2_10Y_Prepared.csv', '..\\prepared_data\\DBS_2SMUE_10Y_Prepared.csv',
    #          '..\\prepared_data\\DBS_APSMA_10Y_Prepared.csv', '..\\prepared_data\\DBS_JET_10Y_Prepared.csv',
    #          '..\\prepared_data\\DBS_KP_10Y_Prepared.csv', '..\\prepared_data\\DBS_SB000_10Y_Prepared.csv',
    #          '..\\prepared_data\\DBS_CIRSC_10Y_Prepared.csv', '..\\prepared_data\\DBS_ACOEM_10Y_Prepared.csv',
    #          '..\\prepared_data\\DBS_UNICC_10Y_Prepared.csv', '..\\prepared_data\\DBS_UPO15_10Y_Prepared.csv']
    #
    # for i in paths:
    #     print(i)
    #     SOURCE_PATH = i
    #     main()

    main()
