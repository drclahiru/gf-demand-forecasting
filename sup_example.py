import pandas as pd
import numpy as np


def exponential_smoothing(x, alpha, s=None):
    if s is None:
        s = []

    if len(s) == 0:
        s.append(x)
    else:
        st = alpha * x + (1 - alpha) * s[-1]
        s.append(st)


def main():
    data = pd.read_excel("C:\\Users\\Goshko\\Desktop\\ops\\ALPHS_FC.xlsx",
                         sheet_name='FC 3 mth horizon',
                         header=1)
    print("Material groups")
    for mg in set(data['Material Group']):
        count = np.sum(data['Material Group'].values == mg)
        print(mg, count)
    filtered_data = data[(data['Material Group'] == 'ALPHS') & (data['Company code'] == 'BGE')]
    print()
    print("Filtered Data")
    print(filtered_data)

    X = np.round(filtered_data['ZPC'].values)

    s = []
    alpha = 0.18  # a number between 0 and 1
    for x in X:
        exponential_smoothing(x, alpha, s)
    print()
    print('real', '*grundfos fc*', 'our fc')
    for a, b, c in zip(filtered_data['ZPC'].values, filtered_data['ZPC.1'].values, s):
        print(int(np.round(a)), int(np.round(b)), int(np.round(c)))
    print()
    print("Forecasting the full sequence (errors)")
    print('*grundfos fc*', 'our fc')
    for a, b, c in zip(filtered_data['ZPC'].values, filtered_data['ZPC.1'].values, s):
        eb = abs(b - a) / a
        ec = abs(c - a) / a
        print(f"{100 * eb:.2f} - {100 * ec:.2f}")

    s = []
    alpha = 0.1  # a number between 0 and 1
    for x in X[:5]:
        exponential_smoothing(x, alpha, s)
    print()
    print("Windowed smoothing")
    s_new = [sn for sn in s]
    print(s_new)

    # updating s_new with real data
    s_new[1] = X[1]
    s_new[2] = X[2]
    s_new[3] = X[3]
    s_new[4] = X[4]

    alpha = 0.1  # a number between 0 and 1
    for x in X[5:8]:
        exponential_smoothing(x, alpha, s_new)
    print()
    print(s_new)


if __name__ == "__main__":
    main()
