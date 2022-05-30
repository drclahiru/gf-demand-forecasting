import warnings
import winsound

import matplotlib.pyplot as plt
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from tensorflow.keras import backend as K

from class_version_code.exec_script_grid_search_lstm import combine_material_groups
from class_version_code.holt import Holts
from class_version_code.lstm import Lstm
from class_version_code.plot_data import PlotData

import time
from datetime import timedelta
start_time = time.monotonic()

# ignoring UserWarnings
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
MODEL = "lstm"

# path from which we extract product group data
SOURCE_PATH = '../prepared_data/DBS_KP_10Y_Prepared.csv'

# parameters for forecasting
TRAINING_SIZE = 102
FORECAST_SIZE = 3

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

    # get the best relative error and the accociated best hyper parameters
    if rel_error < best_rel_error:
        best_rel_error = rel_error
        best_num_of_epoch = num_of_epoch
        best_output_size = output_size
        best_look_back = look_back
        best_learning_rate = learning_rate
        best_activation = activation

        # predictions = [0.0, 0.0, grundfos_forecasting(data, lstm_model, FORECAST_SIZE)]
        # divide the data to be used for plotting
        # lstm_model.divide_data(unit_data)
        # # initialize the data plot class
        # plot_data = PlotData(PLOT_SIZE, lstm_model.train, lstm_model.test, predictions)

        # # plot data with all of the forecasting method/model predictions
        # if MODEL == 'all':
        #     plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months (KP)",
        #                         [holts_model.rel_error, arima_model.rel_error, lstm_model.rel_error])
        # # plot data for one forecasting method/model prediction
        # else:
        #     plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        # plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        # plt.show()

    del lstm_model

    K.clear_session()

    return rel_error


def main():
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI',  # Expected Improvement.
                                n_calls=40, x0=default_parameters)

    print("**************************")
    print("Product group: ", SOURCE_PATH[21:-17])
    print("Forecast size: ", FORECAST_SIZE)
    # print the best relative error
    print("Best relative error:** ", best_rel_error)
    # print the best hyper parameters
    print("The best hyper parameters are: ")
    print("num_of_epoch: ", best_num_of_epoch)
    print("output_size: ", best_output_size)
    print("look_back: ", best_look_back)
    print("learning_rate: ", best_learning_rate)
    print("activation: ", best_activation)

    end_time = time.monotonic()
    print("Duration: ", timedelta(seconds=end_time - start_time))

    print("**************************")
    print("Product group: ", SOURCE_PATH[21:-17])
    print(best_num_of_epoch, best_output_size, best_look_back, best_learning_rate, best_activation)

    frequency = 500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


if __name__ == "__main__":
    main()
