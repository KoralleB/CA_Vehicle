import numpy as np
import pandas as pd


def read_data():
    """
    read three datasets
    :return: three datasets
    """
    veh = pd.read_csv('../data/survey_res_vehicle.csv')
    main = pd.read_csv('../data/survey_res_main.csv')
    per = pd.read_csv('../data/survey_res_person.csv')

    return veh, main, per


def prepro_veh(veh):
    """
    preprocessing vehicle dataset.
    :param veh: vehicle dataset
    :return: preprocessed vehicle dataset
    """
    # shorten column names
    veh = veh.rename(columns={'primary_driver_id': 'perid'})
    veh = veh.rename(columns={'annual_mileage': 'ann_mile'})

    # response variable
    # Recode gas as 0, electric as 1
    veh['y'] = 2
    veh['y'] = np.where(veh.fuel_clean == 1, 0, veh['y'])  # ICE-G
    veh['y'] = np.where(veh.fuel_clean == 2, 0, veh['y'])  # HEV
    veh['y'] = np.where(veh.fuel_clean == 3, 1, veh['y'])  # PHEV
    veh['y'] = np.where(veh.fuel_clean == 4, 0, veh['y'])  # ICE-D
    veh['y'] = np.where(veh.fuel_clean == 5, 1, veh['y'])  # BEV
    veh['y'] = np.where(veh.fuel_clean == 6, 1, veh['y'])  # FCEV
    veh['y'] = np.where(veh.fuel_clean == 7, 1, veh['y'])  # PFCEV
    veh['y'] = np.where(veh.fuel_clean == 8, 0, veh['y'])  # ICE-E85
    veh['y'] = np.where(veh.fuel_clean == 9, 2, veh['y'])  # CNG -> drop
    veh.drop(veh[veh['y'] == 2].index, inplace=True)

    return veh


def prepro_main(main):
    """
    preprocessing household dataset.
    :param main: household dataset
    :return: preprocessed household dataset
    """
    # shorten column names
    main = main.rename(columns={'household_members_4': 'hh_emp'})
    main = main.rename(columns={'home_electricity_access': 'elec_acc'})
    main = main.rename(columns={'tot_hh_members': 'hh_size'})
    main = main.rename(columns={'num_hh_vehicles': 'hh_veh'})

    # household income
    main['hh_inc'] = np.nan
    main['hh_inc'] = np.where(main.income == 1, 1, main['hh_inc'])  # $49,999 or under
    main['hh_inc'] = np.where(main.income == 2, 1, main['hh_inc'])  # $49,999 or under
    main['hh_inc'] = np.where(main.income == 3, 1, main['hh_inc'])  # $49,999 or under
    main['hh_inc'] = np.where(main.income == 4, 1, main['hh_inc'])  # $49,999 or under
    main['hh_inc'] = np.where(main.income == 5, 2, main['hh_inc'])  # $50,000 to $99,999
    main['hh_inc'] = np.where(main.income == 6, 2, main['hh_inc'])  # $50,000 to $99,999
    main['hh_inc'] = np.where(main.income == 7, 3, main['hh_inc'])  # $100,000 to $149,999
    main['hh_inc'] = np.where(main.income == 8, 4, main['hh_inc'])  # $150,000 to $199,999
    main['hh_inc'] = np.where(main.income == 9, 5, main['hh_inc'])  # $200,000 to $249,999
    main['hh_inc'] = np.where(main.income == 10, 6, main['hh_inc'])  # $250,000 or more

    return main


def prepro_per(per):
    """
    preprocessing household member dataset.
    :param per: household member dataset
    :return: preprocessed household member dataset
    """
    # employment
    per['employ'] = np.nan
    per['employ'] = np.where(per.employment == 1, 1, per['employ'])  # employed full-time
    per['employ'] = np.where(per.employment == 2, 1, per['employ'])  # employed part-time
    per['employ'] = np.where(per.employment == 3, 1, per['employ'])  # employed full- and part- time
    per['employ'] = np.where(per.employment == 5, 1, per['employ'])  # employed self
    per['employ'] = np.where(per.employment == 4, 2, per['employ'])  # unemployed

    # student
    per['stu'] = np.nan
    per['stu'] = np.where(per.student == 1, 1, per['stu'])  # full-time campus
    per['stu'] = np.where(per.student == 2, 1, per['stu'])  # part-time campus
    per['stu'] = np.where(per.student == 3, 2, per['stu'])  # online
    per['stu'] = np.where(per.student == 4, 3, per['stu'])  # no

    # race
    per['race'] = np.nan
    per['race'] = np.where(per.ethnicity == 1, 1, per['race'])  # hispanic
    per['race'] = np.where(per.race_1 == 1, 2, per['race'])  # native
    per['race'] = np.where(per.race_2 == 1, 3, per['race'])  # asian
    per['race'] = np.where(per.race_3 == 1, 4, per['race'])  # african american
    per['race'] = np.where(per.race_4 == 1, 5, per['race'])  # pacific islands
    per['race'] = np.where(per.race_5 == 1, 6, per['race'])  # white

    # number of drivers in a household
    temp = per.groupby(['sampno', 'license']).size()
    temp = temp.to_frame(name='hh_drv').reset_index()
    temp.drop(temp[temp['license'] == 2].index, inplace=True)
    per = pd.merge(per, temp[['sampno', 'hh_drv']], on='sampno')

    return per


def merge_hhveh(veh, main, per):
    """
    merge three datasets into one.
    :param veh: vehicle dataset
    :param main: household dataset
    :param per: household member dataset
    :return: merged dataset
    """
    # merge
    data_temp = pd.merge(veh[['sampno', 'perid', 'y', 'ann_mile', 'tnc_veh', 'delivery']],
                         main[['sampno', 'county', 'region', 'hh_veh', 'hh_size', 'hh_emp', 'charge_work',
                               'elec_acc', 'housing', 'hh_inc']], on='sampno')

    data = pd.merge(data_temp,
                    per[['perid', 'gender', 'employ', 'stu', 'hh_drv', 'drive_freq', 'race']], on='perid')

    # replace blank with NaN
    data = data.replace(r'^\s*$', np.nan, regex=True)

    # change column type
    obj_colnames = ['tnc_veh', 'delivery', 'county', 'region', 'charge_work', 'elec_acc', 'housing',
                    'hh_inc', 'gender', 'employ', 'stu', 'drive_freq', 'race']

    for col in obj_colnames:
        data[col] = data[col].astype('object')

    data['ann_mile'] = data['ann_mile'].astype('float64')

    return data


def get_data():
    """
    call for all the functions above to create a single dataframe for modeling.
    """
    veh, main, per = read_data()
    veh = prepro_veh(veh)
    main = prepro_main(main)
    per = prepro_per(per)
    data = merge_hhveh(veh, main, per)
    data.to_csv('../output_files/data.csv', index=False)
