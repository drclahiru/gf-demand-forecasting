import datetime

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

# needed for ets and holt's
WINDOW_SIZE = 14
SMOOTH_LVL = .5
# only needed for holt's
SMOOTH_SLOPE = .25
# only needed for holt's damped method
DAMPED_TREND = .88

COMPANY_CODE = 'BGE'
MATERIAL_GROUP = 'ALPHS'

# ets or holts or holts_damped or holts_winters
FORECAST_TYPE = 'holts_winters'


def holts_winters_forecasting(unit_data):
    train = unit_data.iloc[:WINDOW_SIZE]
    train.index = pd.to_datetime(train.index)

    model = HWES(train.values, trend='mul', seasonal='mul', seasonal_periods=12)
    model._index = pd.to_datetime(train.index)

    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.forecast(len(unit_data) - WINDOW_SIZE + 1)
    return predictions, model_fit


def holts_dampening_forecasting(unit_data):
    predictions = []
    model_fit = None
    for i in range(len(unit_data) - WINDOW_SIZE + 1):
        train = unit_data.iloc[i:WINDOW_SIZE + i]
        train.index = pd.to_datetime(train.index)

        model = Holt(train.values, damped_trend=True)
        model._index = pd.to_datetime(train.index)

        model_fit = model.fit(smoothing_level=SMOOTH_LVL, smoothing_trend=SMOOTH_SLOPE, damping_trend=DAMPED_TREND)
        pred1 = model_fit.forecast(1)
        predictions.append(pred1)
    print(model_fit.summary())
    return predictions, model_fit


def holts_forecasting(unit_data):
    predictions = []
    model_fit = None
    for i in range(len(unit_data) - WINDOW_SIZE + 1):
        train = unit_data.iloc[i:WINDOW_SIZE + i]
        train.index = pd.to_datetime(train.index)

        model = Holt(train.values)
        model._index = pd.to_datetime(train.index)

        model_fit = model.fit(smoothing_level=SMOOTH_LVL, smoothing_trend=SMOOTH_SLOPE)
        pred1 = model_fit.forecast(1)
        predictions.append(pred1)
    print(model_fit.summary())
    return predictions, model_fit


def ets_forecasting(unit_data):
    predictions = []
    model_fit = None
    for i in range(len(unit_data) - WINDOW_SIZE + 1):
        train = unit_data.iloc[i:WINDOW_SIZE + i]
        train.index = pd.to_datetime(train.index)

        model = SimpleExpSmoothing(train.values)
        model._index = pd.to_datetime(train.index)

        model_fit = model.fit(smoothing_level=SMOOTH_LVL)
        pred1 = model_fit.forecast(1)
        predictions.append(pred1)
    print(model_fit.summary())
    return predictions, model_fit


def plot_data(train, test, predictions, fit, unit_max):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, color="gray")
    ax.plot(train.index, train.values, color='gray')
    ax.plot(test.index, predictions, label="alpha=" + str(fit.params['smoothing_level'])[:5], color='#ff7823')
    ax.vlines(train.index[-1], 0, unit_max, linestyle='--', color='r',
              label='Start of forecast')
    if FORECAST_TYPE == 'holts':
        plt.title(f"Holts Smoothing (window size = {WINDOW_SIZE}"
                  f" / smoothing trend = {fit.params['smoothing_trend']})")
    elif FORECAST_TYPE == 'holts_damped':
        plt.title(f"Holts Damped Smoothing (window size = {WINDOW_SIZE}"
                  f" / smoothing trend = {fit.params['smoothing_trend']}"
                  f" / damping trend = {fit.params['damping_trend']})")
    elif FORECAST_TYPE == 'holts_winters':
        plt.title(f"Holts Damped Smoothing (window size = {WINDOW_SIZE}"
                  f" / smoothing trend = {round(fit.params['smoothing_trend'], 4)}"
                  f" / damping seasonal = {round(fit.params['smoothing_seasonal'], 4)})")
    else:
        plt.title(f"Simple Exponential Smoothing (window size = {WINDOW_SIZE})")
    plt.legend()


def main():
    # initialization
    plt.close("all")
    data = pd.read_excel("data\\ALPHS_FC.xlsx",
                         sheet_name='FC 3 mth horizon',
                         header=1)
    total_counter = 0
    correct_counter = 0
    for company_code in set(data['Company code']):
        # Go thorough all of the company codes or just the one set in the global variable "COMPANY_CODE"
        if (COMPANY_CODE is not None and company_code == COMPANY_CODE) or COMPANY_CODE is None:
            print(company_code)
            for material_group in set(data[data['Company code'] == company_code]['Material Group']):
                # Go thorough all of the material groups or just the one set in the global variable "MATERIAL_GROUP"
                if (MATERIAL_GROUP is not None and material_group == MATERIAL_GROUP) or MATERIAL_GROUP is None:
                    # filter the data and convert the dates
                    filtered_data = data[
                        (data['Material Group'] == material_group) & (data['Company code'] == company_code)]
                    months = ["%.4f" % f for f in filtered_data["Cal. year / month"].values]
                    total_counter += 1
                    # only use the combination that have all of the dates (from 09/2018 until 01/2021)
                    if len(months) == 27:
                        correct_counter += 1
                        print(f"\t{material_group} {len(months)}")
                        # generate a time series
                        months = [str(m).replace('.', '/') for m in months]
                        index = pd.date_range(start=months[0], end=months[-1], freq="M")
                        unit_data = pd.Series(filtered_data["ZPC"].values[:-1], index)
                        unit_max = max(filtered_data["ZPC"].values)
                        # execute the exponential/holts smoothing forecast
                        if FORECAST_TYPE == 'holts':
                            predictions, fit = holts_forecasting(unit_data)
                        elif FORECAST_TYPE == 'holts_damped':
                            predictions, fit = holts_dampening_forecasting(unit_data)
                        elif FORECAST_TYPE == 'holts_winters':
                            predictions, fit = holts_winters_forecasting(unit_data)
                        else:
                            predictions, fit = ets_forecasting(unit_data)
                        # seperate the inital train data and the comparison test data from the total unit data
                        train = unit_data.iloc[:WINDOW_SIZE]
                        train.index = pd.to_datetime(train.index)
                        test = unit_data.iloc[WINDOW_SIZE - 1:]
                        test.index = pd.to_datetime(test.index)
                        # plot the results
                        plot_data(train, test, predictions, fit, unit_max)
                        plt.show()
    # information about the give data (do they have all of the dates)
    print(f"Total number of CC/MG combinations: {total_counter}")
    print(f"Number of correct CC/MG combinations: {correct_counter}")
    print(f"Number of faulty CC/MG combinations: {total_counter - correct_counter}")


if __name__ == "__main__":
    main()
