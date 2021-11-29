import pandas as pd
import csv
import datetime as dt
import math

SOURCE_PATH_1 = 'prepared_data\\DBS_10Y_Prepared.csv'

SOURCE_PATH_2 = 'data\\DBS_10_Monthly.xlsx'
SHEET_NAME = 'PtypeHierarchy'

MONTH_LIMIT = None

START_MONTH = '2011-01'
END_MONTH = '2020-12'

MG_COL_NAME = 'MD Material Group'


def main():
    data_1 = pd.read_csv(SOURCE_PATH_1, header=0)
    data_2 = pd.read_excel(SOURCE_PATH_2,
                           sheet_name=SHEET_NAME,
                           header=0)
    mg_data = data_2["Material Group Lvl 7"].values
    pg_data = data_2["Product Group Lvl 5"].values
    pg_mg_dict = dict()
    for i in range(len(pg_data)):
        if pg_data[i] not in pg_mg_dict:
            pg_mg_dict[str(pg_data[i])] = [str(mg_data[i])]
        else:
            pg_mg_dict[str(pg_data[i])].append(str(mg_data[i]))
    for pg in pg_mg_dict:
        has_mg = False
        for mg in pg_mg_dict[pg]:
            if mg in data_1[MG_COL_NAME].values:
                has_mg = True
                break
        if has_mg:
            prepared_data = open(f"prepared_data\\DBS_{pg}_10Y_Prepared.csv", 'w', encoding='UTF8', newline='')
            alphs_writer = csv.writer(prepared_data)
            compare_months = pd.date_range(START_MONTH, END_MONTH,
                                           freq='MS').strftime("%m-%Y").tolist()
            if MONTH_LIMIT is None:
                header = [MG_COL_NAME] + compare_months
            else:
                header = [MG_COL_NAME] + compare_months[:MONTH_LIMIT]
            alphs_writer.writerow(header)
            for mg in pg_mg_dict[pg]:
                if mg in data_1[MG_COL_NAME].values:
                    alphs_writer.writerow(data_1[data_1[MG_COL_NAME] == mg].values[0])
            prepared_data.close()


if __name__ == "__main__":
    main()
