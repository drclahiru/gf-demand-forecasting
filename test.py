import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import unit
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing


def main():
    plt.close("all")
    data = pd.read_excel('data/ALPHS_FC.xlsx',
                         sheet_name='FC 3 mth horizon',
                         header=1)
    filtered_data = data[(data['Material Group'] == 'ALPHS') & (data['Company code'] == 'BGE')]
    index = pd.date_range(start="10/2018", end="10/2019", freq="M")
    print(filtered_data["ZPC"].values)
    print(filtered_data["Cal. year / month"].values)
    unit_data = pd.Series(filtered_data["ZPC"].values[1:-14], index)
    # ax = unit_data.plot()
    # ax.set_xlabel("Months")
    # ax.set_ylabel("ZPC")
    # ax.set_title("Number of req units for ALPHS MG and BGE Company Code")
    # plt.figure(figsize=(12, 8))
    # plt.show()

    fit3 = SimpleExpSmoothing(unit_data.values, initialization_method="known", initial_level=1113.547).fit(optimized=False, smoothing_level=0.2)
    f_cast3 = fit3.forecast(10)
    print()
    print(unit_data)
    print()
    print(f_cast3)
    exit(1)

    fig = plt.figure(figsize=(15, 5), facecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(unit_data, marker="o", color="black", label="Real Data")

    ax.plot(fit3.fittedvalues, marker="o", color="green", label=f"ETS {f_cast3.name}")
    ax.legend()
    ax.set_title("Simple Exponential Smoothing (ALPHS/BGE)")
    ax.set_xlabel("Months")
    ax.set_ylabel("ZPC (Number of Units)")
    plt.show()


if __name__ == "__main__":
    main()
