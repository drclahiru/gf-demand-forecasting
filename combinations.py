import datetime

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

WINDOW_SIZE = 10
SMOOTH_LVL = .8

COMPANY_CODE = 'BGE'
MATERIAL_GROUP = 'ALPHS'


def ets_forecasting(unit_data):
    preds = []
    for i in range(len(unit_data) - WINDOW_SIZE + 1):
        train = unit_data.iloc[i:WINDOW_SIZE + i]
        train.index = pd.to_datetime(train.index)

        model = SimpleExpSmoothing(train.values)
        model._index = pd.to_datetime(train.index)

        fit1 = model.fit(smoothing_level=SMOOTH_LVL)
        pred1 = fit1.forecast(1)
        preds.append(pred1)
    return preds, fit1


def plot_data(train, test, preds, fit, unit_max):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, color="gray")
    ax.plot(train.index, train.values, color='gray')
    ax.plot(test.index, preds, label="alpha=" + str(fit.params['smoothing_level'])[:3], color='#ff7823')
    ax.vlines(train.index[-1], 0, unit_max, linestyle='--', color='r',
              label='Start of forecast')
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
                        # execute the exponential smoothing forecast
                        preds, fit = ets_forecasting(unit_data)
                        # seperate the inital train data and the comparison test data from the total unit data
                        train = unit_data.iloc[:WINDOW_SIZE]
                        train.index = pd.to_datetime(train.index)
                        test = unit_data.iloc[WINDOW_SIZE - 1:]
                        test.index = pd.to_datetime(test.index)
                        # plot the results
                        plot_data(train, test, preds, fit, unit_max)
                        plt.show()
    # information about the give data (do they have all of the dates)
    print(f"Total number of CC/MG combinations: {total_counter}")
    print(f"Number of correct CC/MG combinations: {correct_counter}")
    print(f"Number of faulty CC/MG combinations: {total_counter - correct_counter}")


if __name__ == "__main__":
    main()
