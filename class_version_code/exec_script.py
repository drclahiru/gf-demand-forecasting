import warnings
import pandas as pd
import os
import statistics as stat

from statsmodels.tools.sm_exceptions import ConvergenceWarning

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

# path from which we extract product group data
SOURCE_PATH = '..\\prepared_data\\DBS_SUPM3_10Y_Prepared.csv'

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
MODEL = "all"

# parameters for forecasting
TRAINING_SIZE = 91
FORECAST_SIZE = 8

# parameters for Holts method
SMOOTH_LVL = .6
SMOOTH_SLOPE = .3
DAMPED_TREND = .8

# parameters for the ARIMA model
AUTO_REGRESSION = 1
DIFFERENCING = 1
MOVING_AVREAGE = 1

# parameters for the LSTM model
NUM_OF_EPOCH = 100
OUTPUT_SIZE = 8
LOOK_BACK = 6

# how much data-points to show on plot before forecasting
PLOT_SIZE = 15

# do we print the method and model summaries as well as their progression
DO_PRINT = False


def grundfos_forecasting(prepared_data, forecasting_class):
    simple_exp_model = Holts(TRAINING_SIZE, FORECAST_SIZE, 0.2, 0, 0, is_damped=False,
                        do_print=DO_PRINT)
    product_group_exp_smoothing_model = forecasting_class
    total_mg_predictions = []

    for material_group in prepared_data[MAT_COL_NAME]:
        filtered_data = prepared_data[prepared_data[MAT_COL_NAME] == material_group].values[0][1:]
        new_index = pd.to_datetime(prepared_data.columns[1:])
        unit_data = pd.Series(filtered_data, new_index)
        simple_exp_model.forecast(unit_data)
        mg_predictions = simple_exp_model.predictions
        total_mg_predictions.append(mg_predictions)

    pg_predictions = []
    for i in range(len(total_mg_predictions[0])):
        temp_sum = 0.0
        for j in range(len(prepared_data[MAT_COL_NAME])):
            temp_sum += total_mg_predictions[j][i]
        pg_predictions.append(temp_sum)

    total_mg_ratio_list = []
    for i in range(len(total_mg_predictions)):
        mg_ratio_list = []
        for j in range(len(total_mg_predictions[i])):
            mg_ratio_list.append(total_mg_predictions[i][j]/pg_predictions[j])
        total_mg_ratio_list.append(mg_ratio_list)

    product_group_data = combine_material_groups(prepared_data)
    product_group_exp_smoothing_model.forecast(product_group_data)
    product_group_predictions = product_group_exp_smoothing_model.predictions

    total_mg_final_predictions = []
    for i in range(len(total_mg_ratio_list)):
        mg_final_pred = []
        for j in range(len(total_mg_ratio_list[i])):
            mg_final_pred.append(total_mg_ratio_list[i][j] * product_group_predictions[j])
        total_mg_final_predictions.append(mg_final_pred)

    total_relative_errors = []
    for mg in total_mg_final_predictions:
        product_group_exp_smoothing_model.predictions = mg
        product_group_exp_smoothing_model.calculate_relative_error()
        total_relative_errors.append(product_group_exp_smoothing_model.rel_error)

    final_relative_error = stat.mean(total_relative_errors)

    total_pg_pred = [0.0] * len(total_mg_final_predictions[0])
    for mg in total_mg_final_predictions:
        total_pg_pred = [x + y for x, y in zip(total_pg_pred, mg)]
    product_group_exp_smoothing_model.predictions = total_pg_pred
    return total_pg_pred


def total_print(holts_model, arima_model, lstm_model, predictions, unit_data):
    """
    Print the whole of the product group data

    Parameters
    ----------
    holts_model : Forecast
        the holts method forecasting class
    arima_model : Forecast
        the ARIMA model forecasting class
    lstm_model : Forecast
        the LSTM model forecasting class
    predictions : list
        the prediction of the forecasting
    unit_data: Series
        the entire time series
    """
    if MODEL == 'holt' or MODEL == 'all':
        holts_model.divide_data(unit_data)
        holts_model.predictions = predictions[0]
        holts_model.calculate_relative_error()
        holts_model.print_result()
    if MODEL == 'arima' or MODEL == 'all':
        arima_model.divide_data(unit_data)
        arima_model.predictions = predictions[1]
        arima_model.calculate_relative_error()
        arima_model.print_result()
    if MODEL == 'lstm' or MODEL == 'all':
        lstm_model.divide_data(unit_data)
        lstm_model.predictions = predictions[2]
        lstm_model.calculate_relative_error()
        lstm_model.print_result()


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


def forecast_data(holts_model, arima_model, lstm_model, unit_data, calc_error=True):
    """
    combine all of the material group unit values of the product group

    Parameters
    ----------
    holts_model : Forecast
        the holts method forecasting class
    arima_model : Forecast
        the ARIMA model forecasting class
    lstm_model : Forecast
        the LSTM model forecasting class
    unit_data : Series
        the entire time series
    calc_error : bool
        a flag that determines if we will calculate the relative error

    Returns
    ----------
    predictions: list
        a list of predictions for each forecasting method/model
    """
    if MODEL == 'arima' or MODEL == 'all':
        arima_model.forecast(unit_data)
        if calc_error:
            arima_model.calculate_relative_error()
            arima_model.print_result()
    if MODEL == 'holt' or MODEL == 'all':
        holts_model.forecast(unit_data)
        if calc_error:
            holts_model.calculate_relative_error()
            holts_model.print_result()
    if MODEL == 'lstm' or MODEL == 'all':
        lstm_model.forecast(unit_data)
        if calc_error:
            lstm_model.calculate_relative_error()
            lstm_model.print_result()
    predictions = [holts_model.predictions, arima_model.predictions, lstm_model.predictions]
    return predictions


def main():
    # read the prepared data from the cvs file
    data = pd.read_csv(SOURCE_PATH, header=0)
    # initialize the forecasting methods/models
    arima_model = Arima(TRAINING_SIZE, FORECAST_SIZE, AUTO_REGRESSION, DIFFERENCING, MOVING_AVREAGE, do_print=DO_PRINT)
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
    elif MAT_GROUP == 'Grundfos':
        predictions = [grundfos_forecasting(data, arima_model), grundfos_forecasting(data, holts_model),
                       grundfos_forecasting(data, lstm_model)]
        unit_data = combine_material_groups(data)
    # forecast for a single material group
    else:
        filtered_data = data[data[MAT_COL_NAME] == MAT_GROUP].values[0][1:]
        new_index = pd.to_datetime(data.columns[1:])
        unit_data = pd.Series(filtered_data, new_index)
        predictions = forecast_data(holts_model, arima_model, lstm_model, unit_data)
    # divide the data to be used for plotting
    holts_model.divide_data(unit_data)
    # initialize the the data plot class
    plot_data = PlotData(PLOT_SIZE, holts_model.train, holts_model.test, predictions)
    # plot data with all of the forecasting method/model predictions
    if MODEL == 'all':
        plot_data.plot_data(f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
    # plot data for one forecasting method/model prediction
    else:
        plot_data.plot_one_method(MODEL, f"{MAT_GROUP} Forecast for {FORECAST_SIZE} Months")
    plt.show()


if __name__ == "__main__":
    main()
