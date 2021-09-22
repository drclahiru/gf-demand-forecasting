import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing


def main():
    plt.close("all")
    data = pd.read_excel('data/ALPHS_FC.xlsx',
                         sheet_name='FC 3 mth horizon',
                         header=1)
    for company_code in set(data['Company code']):
        for material_group in set(data[data['Company code'] == company_code]['Material Group']):
            filtered_data = data[(data['Material Group'] == material_group) & (data['Company code'] == company_code)]
            months = ["%.4f" % f for f in filtered_data["Cal. year / month"].values]
            index = [str(m).replace('.', '/') for m in months]
            #index = pd.date_range(start=months[0], end=months[-1], freq="M")
            unit_data = pd.Series(filtered_data["ZPC"].values, index)

            fit3 = SimpleExpSmoothing(unit_data, initialization_method="estimated").fit()
            f_cast3 = fit3.forecast(3).rename(r"$\alpha=%s$" % round(fit3.model.params["smoothing_level"], 4))

            fig = plt.figure(figsize=(15, 5), facecolor='w')
            ax = fig.add_subplot(111)
            ax.plot(unit_data, marker="o", color="black", label="Real Data")

            ax.plot(fit3.fittedvalues, marker="o", color="green", label=f"ETS {f_cast3.name}")
            ax.legend()
            ax.set_title(f"Simple Exponential Smoothing ({material_group}/{company_code})")
            ax.set_xlabel("Months")
            ax.set_ylabel("ZPC (Number of Units)")
            plt.show()
        break


if __name__ == "__main__":
    main()
