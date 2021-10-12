import pandas as pd
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima.model import ARIMA

SOURCE_PATH = 'prepared_data\\ALPHS_FC_Prepared_15_mth.csv'
MAT_GROUP = "ALPHS"

# holt or arima
MODEL = "arima"

WINDOW_SIZE = 12
FORECAST_SIZE = 3
SMOOTH_LVL = .7
SMOOTH_SLOPE = 0.001
DAMPED_TREND = 0.98

AUTO_REGRESSION = 1
DIFFERENCING = 1
MOVING_AVREAGE = 1


def arima_forecasting_3_mnt(unit_data):
    train = unit_data.iloc[:WINDOW_SIZE]
    train = train.asfreq('MS')
    train = train.astype('int')
    model = ARIMA(train.values, order=(AUTO_REGRESSION, DIFFERENCING, MOVING_AVREAGE))
    model_fit = model.fit()
    predictions = model_fit.forecast(3)
    print(model_fit.summary())
    return predictions


def holts_dampening_forecasting_3_mnt(unit_data):
    train = unit_data.iloc[:WINDOW_SIZE]
    train = train.asfreq('MS')
    model = Holt(train.values, damped_trend=True)
    model._index = pd.to_datetime(train.index)
    model_fit = model.fit(smoothing_level=SMOOTH_LVL, smoothing_trend=SMOOTH_SLOPE, damping_trend=DAMPED_TREND)
    predictions = model_fit.forecast(FORECAST_SIZE)
    print(model_fit.summary())
    return predictions


def calculate_relative_error(predictions, test):
    forecast_errors = [abs(test[i] - predictions[i]) / max(test[i], predictions[i]) for i in range(len(test))]
    bias = sum(forecast_errors) * 1.0 / len(test)
    return bias


def main():
    # initialization
    data = pd.read_csv(SOURCE_PATH, header=0)
    filtered_data = data[data["Material Group"] == "ALPHS"].values[0][1:]
    new_index = pd.to_datetime(data.columns[1:])
    unit_data = pd.Series(filtered_data, new_index)
    if MODEL == 'arima':
        predictions = arima_forecasting_3_mnt(unit_data)
    else:
        predictions = holts_dampening_forecasting_3_mnt(unit_data)
    print()
    print(f"Predictions: \n{predictions}")
    test = unit_data.iloc[WINDOW_SIZE:]
    test.index = pd.to_datetime(test.index)
    rel_error = calculate_relative_error(predictions, test.values)
    print()
    print(f"Relative Error: {rel_error}")


if __name__ == "__main__":
    main()
