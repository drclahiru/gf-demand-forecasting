import math
import warnings
import pandas as pd
import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import statistics as stat

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from class_version_code.exec_script_grid_search_holts import combine_material_groups, forecast_data, total_print, \
    grundfos_forecasting

# Ignoring warnings and info logs from tensorflow
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
SOURCE_PATH = '..\\prepared_data\\DBS_APSMA_10Y_Prepared.csv'

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

# parameters for forecasting
TRAINING_SIZE = 102
FORECAST_SIZE = 18

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


def run(smooth_lvl, smooth_slope, damped_trend, forecast_size):
    global grid_search_predictions
    # read the prepared data from the cvs file
    data = pd.read_csv(SOURCE_PATH, header=0)
    # initialize the forecasting methods/models
    arima_model = Arima(TRAINING_SIZE, FORECAST_SIZE, AUTO_REGRESSION, DIFFERENCING, MOVING_AVREAGE, do_print=DO_PRINT)
    holts_model = Holts(TRAINING_SIZE, forecast_size, smooth_lvl, smooth_slope, damped_trend, is_damped=True,
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
        predictions = [grundfos_forecasting(data, holts_model, forecast_size), 0.0, 0.0]
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
    return holts_model.rel_error


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
    global bayesian_predictions

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_smooth_lvl = smooth_lvl
        best_smooth_slope = smooth_slope
        best_damped_trend = damped_trend

        bayesian_predictions = grundfos_forecasting(data, holts_model, FORECAST_SIZE)
        predictions = [0.0, grid_search_predictions[0], bayesian_predictions]
        # predictions = [grundfos_forecasting(data, holts_model, FORECAST_SIZE), 0.0, 0.0]
        # divide the data to be used for plotting
        holts_model.divide_data(unit_data)
        # initialize the data plot class
        plot_data = PlotData(PLOT_SIZE, holts_model.train, holts_model.test, predictions)
        #
        # # plot data with all of the forecasting method/model predictions
        # if MODEL == 'all':
        #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
        #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
        # # plot data for one forecasting method/model prediction
        # else:
        #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        plot_data.plot_optimization(f"Holt's {FORECAST_SIZE} months forecast for {SOURCE_PATH[21:-17]} product group")
        plt.show()

    return rel_error


def grid_search():
    # Grid Search for best parameters for a single horizon
    # choose the horizon by changing the FORECAST_SIZE variable
    best_rel_error = [2.0, 0.0, 0.0, 0.0]
    for smooth_lvl in tqdm(np.arange(0.1, 1.0, 0.1)):
        for smooth_slope in np.arange(0.1, 1.0, 0.1):
            for damped_trend in np.arange(0.1, 1.0, 0.1):
                temp_rel_error = run(smooth_lvl, smooth_slope, damped_trend, FORECAST_SIZE)
                if temp_rel_error < best_rel_error[0]:
                    best_rel_error = [temp_rel_error, smooth_lvl, smooth_slope, damped_trend]


def main():
    grid_search()
    gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=100, x0=default_parameters)


if __name__ == "__main__":
    main()
