import matplotlib.pyplot as plt
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

from class_version_code.holt import Holts
from class_version_code.lstm import Lstm
from class_version_code.plot_data import PlotData

# path from which we extract product group data
SOURCE_PATH = '../prepared_data/DBS_2SMUE_10Y_Prepared.csv'

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

dim_num_of_epoch = Integer(low=2, high=100, name='num_of_epoch')
dim_output_size = Integer(low=1, high=20, name='output_size')
dim_look_back = Integer(low=5, high=101, name='look_back')
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_activation = Categorical(categories=['tanh', 'sigmoid'], name='activation')

dimensions = [dim_num_of_epoch, dim_output_size, dim_look_back, dim_learning_rate, dim_activation]

default_parameters = [2, 1, 5, 1e-6, 'tanh']
best_rel_error = 1.0


def grundfos_forecasting(prepared_data, product_group_exp_smoothing_model, forecast_size):
    simple_exp_model = Holts(TRAINING_SIZE, forecast_size, 0.2, 0, 0, is_damped=False,
                             do_print=DO_PRINT)
    # execute simple exponential smoothing on all material groups
    total_mg_predictions = []
    for material_group in prepared_data[MAT_COL_NAME]:
        filtered_data = prepared_data[prepared_data[MAT_COL_NAME] == material_group].values[0][1:]
        new_index = pd.to_datetime(prepared_data.columns[1:])
        unit_data = pd.Series(filtered_data, new_index)
        simple_exp_model.forecast(unit_data)
        mg_predictions = simple_exp_model.predictions
        total_mg_predictions.append(mg_predictions)
    # sum up the material group predictions
    pg_predictions = []
    for i in range(len(total_mg_predictions[0])):
        temp_sum = 0.0
        for j in range(len(prepared_data[MAT_COL_NAME])):
            temp_sum += total_mg_predictions[j][i]
        pg_predictions.append(temp_sum)
    # create the ratios for every material group withing the product group
    total_mg_ratio_list = []
    for i in range(len(total_mg_predictions)):
        mg_ratio_list = []
        for j in range(len(total_mg_predictions[i])):
            if pg_predictions[j] == 0.0:
                mg_ratio_list.append(0.0)
            else:
                mg_ratio_list.append(total_mg_predictions[i][j] / pg_predictions[j])
        total_mg_ratio_list.append(mg_ratio_list)
    # forecast for the product group
    product_group_data = combine_material_groups(prepared_data)
    product_group_exp_smoothing_model.forecast(product_group_data)
    product_group_predictions = product_group_exp_smoothing_model.predictions
    # use the ratios and the product group redictions to get better material group predictions
    total_mg_final_predictions = []
    for i in range(len(total_mg_ratio_list)):
        mg_final_pred = []
        for j in range(len(total_mg_ratio_list[i])):
            mg_final_pred.append(total_mg_ratio_list[i][j] * product_group_predictions[j])
        total_mg_final_predictions.append(mg_final_pred)
    # sum up all of the final material group predictions to get the product group predictions
    total_pg_final_prediction = []
    for i in range(len(total_mg_final_predictions[0])):
        temp_sum = 0.0
        for j in range(len(prepared_data[MAT_COL_NAME])):
            temp_sum += total_mg_final_predictions[j][i]
        total_pg_final_prediction.append(temp_sum)
    # calcualte the relative error
    product_group_exp_smoothing_model.predictions = total_pg_final_prediction
    product_group_exp_smoothing_model.calculate_relative_error()

    return total_pg_final_prediction


def combine_material_groups(data):
    """
    combine all of the material group unit values of the product group

    Parameters
    ----------
    data : Series
        the entire dataset separated by material groups

    Returns
    ----------
    unit_data: Series
        the summed up product group time series
    """
    total_unit_data_values = None
    # sum up the unit numbers for each material group
    for mat_group in data[MAT_COL_NAME]:
        filtered_data = data[data[MAT_COL_NAME] == mat_group].values[0][1:]
        if total_unit_data_values is None:
            # initalize the total unit number list
            total_unit_data_values = [0.0] * len(filtered_data)
        total_unit_data_values = [x + y for x, y in zip(total_unit_data_values, filtered_data)]
    new_index = pd.to_datetime(data.columns[1:])
    # create the final time series
    unit_data = pd.Series(total_unit_data_values, new_index)
    return unit_data


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
    print('num_of_epoch:', num_of_epoch)
    print('output_size:', output_size)
    print('look_back:', look_back)
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('activation:', activation)
    print()

    # initialize the forecasting model
    lstm_model = Lstm(TRAINING_SIZE, FORECAST_SIZE, num_of_epoch, output_size, look_back, learning_rate=learning_rate,
                      activation=activation, do_print=DO_PRINT)
    data = pd.read_csv(SOURCE_PATH, header=0)
    unit_data = combine_material_groups(data)
    lstm_model.forecast(unit_data)
    lstm_model.calculate_relative_error()
    rel_error = lstm_model.rel_error

    print("Relative error:" + str(rel_error) + "\n")

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

        predictions = [0.0, 0.0, grundfos_forecasting(data, lstm_model, FORECAST_SIZE)]
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
        plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
        plt.show()

    return rel_error


def main():
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI',  # Expected Improvement.
                                n_calls=20, x0=default_parameters)

    # print the best relative error
    print("Best relative error: ", best_rel_error)
    # print the best hyper parameters
    print("The best hyper parameters are: ")
    print("num_of_epoch: ", best_num_of_epoch)
    print("output_size: ", best_output_size)
    print("look_back: ", best_look_back)
    print("learning_rate: ", best_learning_rate)
    print("activation: ", best_activation)


if __name__ == "__main__":
    main()
