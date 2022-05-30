import math
import warnings
import pandas as pd
import os
import numpy as np
import statistics as stat

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from tensorflow.keras import backend as K

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from class_version_code.exec_script_grid_search_lstm import forecast_data, combine_material_groups, total_print, \
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
MODEL = "lstm"

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
dim_num_of_epoch = Integer(low=1, high=100, name='num_of_epoch')
dim_output_size = Integer(low=1, high=20, name='output_size')
dim_look_back = Integer(low=10, high=101, name='look_back')
dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_activation = Categorical(categories=['tanh', 'sigmoid'], name='activation')

dimensions = [dim_num_of_epoch, dim_output_size, dim_look_back, dim_learning_rate, dim_activation]

default_parameters = [1, 1, 10, 1e-6, 'tanh']
best_rel_error = 1.0


def run(num_of_epoch, output_size, look_back, forecast_size):
    global grid_search_predictions
    # read the prepared data from the cvs file
    data = pd.read_csv(SOURCE_PATH, header=0)
    # initialize the forecasting methods/models
    arima_model = Arima(TRAINING_SIZE, FORECAST_SIZE, AUTO_REGRESSION, DIFFERENCING, MOVING_AVREAGE, do_print=DO_PRINT)
    holts_model = Holts(TRAINING_SIZE, FORECAST_SIZE, SMOOTH_LVL, SMOOTH_SLOPE, DAMPED_TREND, is_damped=True,
                        do_print=DO_PRINT)
    lstm_model = Lstm(TRAINING_SIZE, forecast_size, num_of_epoch, output_size, look_back, do_print=DO_PRINT)
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
        predictions = [0.0, 0.0, grundfos_forecasting(data, lstm_model, forecast_size)]
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
    return lstm_model.rel_error


@use_named_args(dimensions=dimensions)
def fitness(num_of_epoch, output_size, look_back, learning_rate, activation):
    """
    initialize the lstm model and calculate the relative error

    Parameters
    ----------
    num_of_epoch : int
        the number of epochs used to train the neural network
    output_size : int
        dimensionality of the output space for the LSTM layers
    look_back : int
        the number of months that will go as input to the lstm
    learning_rate : float
        learning rate of the lstm model
    activation : str
        activation function of the lstm model

    Returns
    ----------
    rel_error: float
        the relative error for the given set of hyper parameters
    """
    # Print the hyper parameters
    # print('num_of_epoch:', num_of_epoch)
    # print('output_size:', output_size)
    # print('look_back:', look_back)
    # print('learning rate: {0:.1e}'.format(learning_rate))
    # print('activation:', activation)
    # print()

    # initialize the forecasting model
    lstm_model = Lstm(TRAINING_SIZE, FORECAST_SIZE, num_of_epoch, output_size, look_back, learning_rate=learning_rate,
                      activation=activation, do_print=DO_PRINT)
    data = pd.read_csv(SOURCE_PATH, header=0)
    unit_data = combine_material_groups(data)
    lstm_model.forecast(unit_data)
    lstm_model.calculate_relative_error()
    rel_error = lstm_model.rel_error

    # print("Relative error:" + str(rel_error) + "\n")

    global best_rel_error
    global best_num_of_epoch
    global best_output_size
    global best_look_back
    global best_learning_rate
    global best_activation
    global bayesian_predictions

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_num_of_epoch = num_of_epoch
        best_output_size = output_size
        best_look_back = look_back
        best_learning_rate = learning_rate
        best_activation = activation

        bayesian_predictions = grundfos_forecasting(data, lstm_model, FORECAST_SIZE)
        predictions = [0.0, grid_search_predictions[2], bayesian_predictions]
        # divide the data to be used for plotting
        lstm_model.divide_data(unit_data)
        # initialize the data plot class
        plot_data = PlotData(PLOT_SIZE, lstm_model.train, lstm_model.test, predictions)

        # # plot data with all of the forecasting method/model predictions
        # if MODEL == 'all':
        #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
        #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
        # # plot data for one forecasting method/model prediction
        # else:
        #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        plot_data.plot_optimization(f"LSTM {FORECAST_SIZE} months forecast for {SOURCE_PATH[21:-17]} product group")
        plt.show()

    # del lstm_model
    #
    # K.clear_session()

    return rel_error


def grid_search():
    # Grid Search for best parameters for a single horizon
    best_rel_error = [2.0, 0.0, 0.0, 0.0]
    for num_of_epoch in tqdm(np.arange(20, 100, 35)):
        for output_size in np.arange(2, 15, 6):
            for look_back in np.arange(21, 101, 40):
                mean_temp_lstm_rel_error = []
                # run lstm a few times for each parameter combination and them mean the errors to get better results
                for i in range(3):
                    mean_temp_lstm_rel_error.append(run(num_of_epoch, output_size, look_back, FORECAST_SIZE))
                temp_rel_error = np.mean(mean_temp_lstm_rel_error)
                if temp_rel_error < best_rel_error[0]:
                    best_rel_error = [temp_rel_error, num_of_epoch, output_size, look_back]


def main():
    grid_search()
    gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=40, x0=default_parameters)


if __name__ == "__main__":
    main()
