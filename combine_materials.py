import pandas as pd
import csv
import datetime as dt

START_MONTH = '2018-10'
END_MONTH = '2020-12'

SC_COL_NAME = 'Company Code'
MG_COL_NAME = 'MD Material Group'
MONTH_COL_NAME = 'Cal. year / month'
UNIT_COL_NAME = 'PC'
MAT_COL_NAME = 'Material'

SOURCE_PATH = 'data\\ALPHS_10Y_ByMaterial.xlsx'
DEST_PATH = 'prepared_data\\ALPHS_10Y_ByMaterial_Prepared.csv'
SHEET_NAME = 'SalesHistory'

MONTH_LIMIT = None


def check_months(filtered_data, compare_months):
    filtered_units = filtered_data[UNIT_COL_NAME].values
    unit_list = [0] * len(compare_months)
    # create a mont list from the filtered data
    my_months = ["%.4f" % f for f in filtered_data[MONTH_COL_NAME].values]
    my_months = [str(m).replace('.', '-') for m in my_months]
    month_list = [dt.datetime.strptime(date, '%m-%Y').date() for date in my_months]
    month_list = [dt.datetime.strftime(date, "%m-%Y") for date in month_list]
    shift = 0
    # to the existing months in the filtered data, enter the unit values in the unit list
    # otherwise leave it at zero
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
    # generate a list of months for the header
    compare_months = pd.date_range(START_MONTH, END_MONTH,
                                   freq='MS').strftime("%m-%Y").tolist()
    # generate the header (material group / material / all of the dates
    if MONTH_LIMIT is None:
        header = [MG_COL_NAME] + [MAT_COL_NAME] + compare_months
    else:
        header = [MG_COL_NAME] + [MAT_COL_NAME] + compare_months[:MONTH_LIMIT]
    alphs_writer.writerow(header)
    # go through all of the material groups
    for material_group in set(data[MG_COL_NAME]):
        # save pairs of materials and their total number of units thorughout the months
        material_units = []
        # go through all of the materials for the MG
        for material in set(data[data[MG_COL_NAME] == material_group][MAT_COL_NAME]):
            # save the sum of all of the units for the specific company code, material nad material group
            total_units = [0] * len(compare_months)
            # go through all of the company codes for the given material nad material group
            for company_code in set(
                    data[(data[MG_COL_NAME] == material_group) & (data[MAT_COL_NAME] == material)][SC_COL_NAME]):
                # extract one point of data
                filtered_data = data[(data[MG_COL_NAME] == material_group) & (data[SC_COL_NAME] == company_code) & (
                        data[MAT_COL_NAME] == material)]
                # get the number of units from the months
                unit_list = check_months(filtered_data, compare_months)
                # sum up the number of gotten units to the total number of units
                # if unit list values is negative, take zero
                total_units = [max(x, 0) + max(y, 0) for x, y in zip(total_units, unit_list)]
            material_units.append([material, total_units])
        # get the old/new material code pairs
        old_new = pd.read_excel('data\\ALPHA2_OldtoNew_PNMapping.xlsx', header=0)
        old_codes = old_new["Old PN"].values.tolist()
        old_codes = [str(x) for x in old_codes]
        new_codes = old_new["New PN"].values.tolist()
        new_codes = [str(x) for x in new_codes]
        # combine old and new material codes to sum up their number of units
        new_materials = []
        for mat in material_units:
            is_found = False
            if mat[0] in old_codes:
                old_index = old_codes.index(mat[0])
                mat = [new_codes[old_index], mat[1]]
            for i in range(len(new_materials)):
                if mat[0] == new_materials[i][0]:
                    new_materials[i][1] = [x + y for x, y in zip(new_materials[i][1], mat[1])]
                    is_found = True
                    break
            if not is_found:
                new_materials.append(mat)


        if MONTH_LIMIT is None:
            for elem in new_materials:
                alphs_writer.writerow([material_group] + [elem[0]] + elem[1])
        else:
            for elem in new_materials:
                alphs_writer.writerow([material_group] + [elem[0]] + elem[1][:MONTH_LIMIT])
    prepared_data.close()


if __name__ == "__main__":
    main()
