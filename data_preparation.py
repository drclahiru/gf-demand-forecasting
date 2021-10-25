import pandas as pd
import csv
import datetime as dt

START_MONTH = '2018-10'
END_MONTH = '2020-12'

SC_COL_NAME = 'Company code'
MG_COL_NAME = 'Material Group'
MONTH_COL_NAME = 'Cal. year / month'
UNIT_COL_NAME = 'ZPC'

SOURCE_PATH = 'data\\ALPHS_FC.xlsx'
DEST_PATH = 'prepared_data\\ALPHS_FC_Prepared_15_mth.csv'
SHEET_NAME = 'FC 3 mth horizon'

MONTH_LIMIT = 15


def check_months(filtered_data, compare_months):
    filtered_units = filtered_data[UNIT_COL_NAME].values
    unit_list = [0] * len(compare_months)
    my_months = ["%.4f" % f for f in filtered_data[MONTH_COL_NAME].values]
    my_months = [str(m).replace('.', '-') for m in my_months]
    month_list = [dt.datetime.strptime(date, '%m-%Y').date() for date in my_months]
    month_list = [dt.datetime.strftime(date, "%m-%Y") for date in month_list]
    shift = 0
    for i in range(len(compare_months)):
        if compare_months[i] in month_list:
            unit_list[i] = int(filtered_units[i - shift])
        else:
            shift += 1
    return unit_list


def main():
    # initialization
    data = pd.read_excel(SOURCE_PATH,
                         sheet_name=SHEET_NAME,
                         header=1)

    prepared_data = open(DEST_PATH, 'w', encoding='UTF8', newline='')
    alphs_writer = csv.writer(prepared_data)
    compare_months = pd.date_range(START_MONTH, END_MONTH,
                                   freq='MS').strftime("%m-%Y").tolist()
    if MONTH_LIMIT is None:
        header = [MG_COL_NAME] + compare_months
    else:
        header = [MG_COL_NAME] + compare_months[:MONTH_LIMIT]
    alphs_writer.writerow(header)
    for material_group in set(data[MG_COL_NAME]):
        total_units = [0] * len(compare_months)
        for company_code in set(data[data[MG_COL_NAME] == material_group][SC_COL_NAME]):
            filtered_data = data[(data[MG_COL_NAME] == material_group) & (data[SC_COL_NAME] == company_code)]
            unit_list = check_months(filtered_data, compare_months)
            total_units = [max(x, 0) + max(y, 0) for x, y in zip(total_units, unit_list)]
        if MONTH_LIMIT is None:
            alphs_writer.writerow([material_group] + total_units)
        else:
            alphs_writer.writerow([material_group] + total_units[:MONTH_LIMIT])

    prepared_data.close()


if __name__ == "__main__":
    main()
