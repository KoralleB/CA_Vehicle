import numpy as np
import pandas as pd

def read_data():
    veh = pd.read_excel('../data/CA_Vehicle.xlsx')
    hh = pd.read_excel('../data/CA_Household.xlsx')
    return veh, hh


def prepro_veh(veh):
    veh = veh.loc[~(veh['vehyear'] < 2011), :]  # remove vehicles older than 6 years

    # subset car, suv
    type_rm = [2, 4, 5, 6, 7, 97, -7, -8]
    for i in type_rm:
        veh = veh.loc[~(veh['vehtype'] == i), :]

    # Re-code gas as 0, electric as 1
    veh['fuel'] = 2
    veh['fuel'] = np.where(veh.fueltype == 1, 0, veh['fuel'])  # ICE-G
    veh['fuel'] = np.where(veh.fueltype == 2, 0, veh['fuel'])  # ICE-D
    veh['fuel'] = np.where(np.logical_and(veh.fueltype == 3, veh.hfuel == 2), 1, veh['fuel'])  # 'PHEV'
    veh['fuel'] = np.where(np.logical_and(veh.fueltype == 3, veh.hfuel == 3), 1, veh['fuel'])  # 'BEV'
    veh['fuel'] = np.where(np.logical_and(veh.fueltype == 3, veh.hfuel == 4), 0, veh['fuel'])  # 'HEV'
    veh.drop(veh[veh['fuel'] == 2].index, inplace=True)

    veh['vehmiles'] = veh['vehmiles'].replace(
        {-88: np.nan, -1: np.nan, -77: np.nan, -7: np.nan, -8: np.nan})  # convert all sorts of no-answer to NaN
    return veh


def prepro_hh(hh):
    hh['homeown'] = hh['homeown'].replace({-7: np.nan})  # convert “I prefer not to answer” to NaN

    # household income
    hh['income'] = np.nan
    for i in [1, 2, 3, 4, 5]:
        hh['income'] = np.where(hh.hhfaminc == i, 1, hh['income'])  # $0 to $49,999
    for i in [6, 7]:
        hh['income'] = np.where(hh.hhfaminc == i, 2, hh['income'])  # $50,000 to $99,999
    for i in [8, 9]:
        hh['income'] = np.where(hh.hhfaminc == i, 3, hh['income'])  # $100,000 to $149,999
    hh['income'] = np.where(hh.hhfaminc == 10, 4, hh['income'])  # $150,000 to $199,999
    hh['income'] = np.where(hh.hhfaminc == 11, 5, hh['income'])  # over $200,000

    # race
    hh['race'] = hh['hh_race']
    hh['race'] = np.where(np.logical_and(hh.hh_race == 97, hh.hh_hisp == 1), 7, hh['race'])  # recode hispanic
    hh['race'] = hh['race'].replace({-7: np.nan})  # convert “I prefer not to answer” to NaN
    hh['race'] = hh['race'].replace({-8: np.nan})  # convert “I dont know” to NaN

    hh['lif_cyc'] = hh['lif_cyc'].replace({-9: np.nan})  # convert “Not Ascertained” to NaN

    hh.drop(hh[hh['drvrcnt'] == 0].index, inplace=True)  # respondent didn't understand the question
    return hh


def hh_loc(hh):
    # remove non-CA
    locs_rm = ['12580', '17980', '19660', '19740', '21660', '22800', '23820', '28140', '29820', '36420', '36500',
               '37980', '38060', '39660', '41620', '42660', '46060']
    for loc_rm in locs_rm:
        hh = hh.loc[~(hh['hh_cbsa'] == loc_rm), :]

    hh.reset_index(drop=True, inplace=True)

    # 6 zones
    california = {'Sac': ['40900', '44700'],
                  'SF': ['41860', '41940', '42100', '42220', '46700', '34900'],
                  'LA': ['31080', '42200', '42020', '37100'],
                  'North': ['17020', '39820', '21700', '46020', '46380', '39780', '17340', '49700', '18860', '45000'],
                  'Central': ['41500', '23420', '12540', '33700', '47300', '32900', '31460', '25260'],
                  'South': ['41740', '40140', '43760', '20940']}

    hh['location'] = np.nan

    for key, value in california.items():
        for zone in value:
            if key == 'Sac':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'Sac', hh['location'])
            if key == 'SF':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'SF', hh['location'])
            if key == 'LA':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'LA', hh['location'])
            if key == 'North':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'North', hh['location'])
            if key == 'Central':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'Central', hh['location'])
            if key == 'South':
                hh['location'] = np.where(hh.hh_cbsa == zone, 'South', hh['location'])

    return hh


def merge_hhveh(veh, hh):
    data = pd.merge(veh[['houseid', 'fuel', 'vehmiles']],
                    hh[['houseid', 'homeown', 'income', 'race', 'hhvehcnt', 'hhsize', 'numadlt', 'drvrcnt', 'wrkcount',
                        'lif_cyc', 'urbrur', 'htppopdn', 'location']], on='houseid')

    # categorical columns
    obj_colnames = ['homeown', 'income', 'race', 'lif_cyc', 'urbrur', 'htppopdn']
    for col in obj_colnames:
        data[col] = data[col].astype('object')
    return data


def get_data():
    veh, hh = read_data()
    veh = prepro_veh(veh)
    hh = prepro_hh(hh)
    hh = hh_loc(hh)
    data = merge_hhveh(veh, hh)
    data.to_csv('../output_files/data.csv',index=False)

