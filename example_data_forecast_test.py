import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

EXAMPLE_DATAS = [("daily-min-temperatures-1981-1990.csv", 'D'), ("daily-total-female-births-1959.csv", 'D'),
                 ("monthly-sunspots-1784-1993.csv", 'MS'), ("retail_sales_used_car_dealers_us_1992_2020.csv", 'MS')]

WINDOW_SIZE = 10
FORECAST_SIZE = 3
SMOOTH_LVL = 0.2
SMOOTH_SLOPE = .7
DAMPED_TREND = .88


def holts_dampening_forecasting_3_mnt(unit_data):
    model_fit = None
    predictions = []
    for j in range(FORECAST_SIZE):
        temp = unit_data.iloc[:WINDOW_SIZE + j]
        train = temp.copy()
        train.index = pd.to_datetime(train.index)
        for k in range(j):
            train.values[-(k + 1)] = predictions[-(k + 1)]
        model = Holt(train.values, damped_trend=True)
        model._index = pd.to_datetime(train.index)

        model_fit = model.fit(smoothing_level=SMOOTH_LVL, smoothing_trend=SMOOTH_SLOPE, damping_trend=DAMPED_TREND)
        pred1 = model_fit.forecast(1)
        predictions.append(pred1)
    print(model_fit.summary())
    return predictions, model_fit


def calculate_relative_error(predictions, test):
    forecast_errors = [abs(test[i] - predictions[i]) / max(test[i], predictions[i]) for i in range(len(test))]
    bias = sum(forecast_errors) * 1.0 / len(test)
    return bias


def plot_data(train, test, predictions, fit, unit_max):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, color="gray")
    ax.plot(train.index, train.values, color='gray')
    ax.plot(test.index, predictions, label="alpha=" + str(fit.params['smoothing_level'])[:5], color='#ff7823')
    ax.vlines(train.index[-1], 0, unit_max, linestyle='--', color='r',
              label='Start of forecast')
    plt.title(f"Holts Damped Smoothing (window size = {WINDOW_SIZE}"
              f" / smoothing trend = {fit.params['smoothing_trend']}"
              f" / damping trend = {fit.params['damping_trend']})")
    plt.legend()


def main():
    plt.close("all")
    example_data = EXAMPLE_DATAS[3]
    series = pd.read_csv(f"example_data\\{example_data[0]}", header=0, parse_dates=[0], index_col=0, squeeze=True,
                         dayfirst=True)
    series = series.asfreq(example_data[1])
    # series.index.freq = example_data[1]
    predictions, fit = holts_dampening_forecasting_3_mnt(series)
    train = series.iloc[:WINDOW_SIZE]
    train.index = pd.to_datetime(train.index)
    test = series.iloc[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + FORECAST_SIZE]
    test.index = pd.to_datetime(test.index)
    rel_error = calculate_relative_error(predictions, test.values)
    print(f"Relative Error = {rel_error}")
    # plot the results
    max_value = max(train.values)
    plot_data(train, test, predictions, fit, max_value)
    plt.show()


if __name__ == "__main__":
    main()
