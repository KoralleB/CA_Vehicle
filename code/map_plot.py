import pandas as pd
import numpy as np
import math
from plotly.offline import init_notebook_mode
from urllib.request import urlopen
import json
import plotly.express as px

init_notebook_mode(connected=True)


def read_data(model_pred=None):
    data = pd.read_csv('../output_files/data.csv')
    X_test = pd.read_csv('../output_files/X_test.csv')

    data = data[data.index.isin(X_test.index)]

    # replace number with name
    region_names = ['Central', 'LA', 'Rest', 'SD', 'SF', 'Sac']
    for num, name in enumerate(region_names, start=1):
        data.region = data.region.replace(num, name)

    if model_pred == 'rf':
        y_pred = np.load('../output_files/y_pred_rf.npy')
        data['y_pred'] = y_pred

    if model_pred == 'logit':
        y_pred = np.load('../output_files/y_pred_log.npy')
        data['y_pred'] = y_pred

    data.drop(data[data['region'] == 7].index, inplace=True)  # drop "I don't know"

    return data


def ca_county():
    df_county = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
    df_county = df_county[df_county['STNAME'] == 'California']
    df_county = df_county.loc[:, df_county.columns.isin(['FIPS', 'STNAME', 'CTYNAME'])]

    # 6 zones
    california_counties = {'Central': [6019, 6029, 6031, 6039, 6047, 6077, 6099, 6107],
                           'LA': [6025, 6037, 6059, 6065, 6071, 6111],
                           'Rest': [6003, 6005, 6007, 6009, 6011, 6015, 6021, 6023, 6027, 6033, 6035, 6043, 6045, 6049,
                                    6051, 6053, 6057, 6063, 6069, 6079, 6083, 6087, 6089, 6091, 6093, 6103, 6105, 6109],
                           'SD': [6073],
                           'SF': [6001, 6013, 6041, 6055, 6075, 6081, 6085, 6095, 6097],
                           'Sac': [6017, 6061, 6067, 6101, 6113, 6115]
                           }

    df_county['region'] = np.nan

    for key, value in california_counties.items():
        for zone in value:
            df_county['region'] = np.where(df_county.FIPS == zone, key, df_county['region'])

    return df_county


def design_county(data, df_county, ycolumn):
    """
    :param data:
    :param df_county:
    :param ycolumn: 'y' or 'y_pred'
    :return:
    """
    # region_names = ['Central', 'LA', 'Rest', 'SD', 'SF', 'Sac']
    # temp = data.groupby(ycolumn)['region'].value_counts().sort_index()
    # region lists per vehicle type
    # done in case some regions have 0 vehicles predicted
    temp = data.copy()
    temp['ones'] = 1
    temp = temp.groupby([ycolumn, 'region'])['ones'].sum().to_frame('count').reset_index()

    reg_list0 = []
    reg_list1 = []
    for i in range(len(temp)):
        if temp[ycolumn][i] == 0:
            reg_list0.append(temp.region[i])
        if temp[ycolumn][i] == 1:
            reg_list1.append(temp.region[i])

    df_county['ICEV'] = np.nan
    df_county['EV'] = np.nan
    df_county['EV_per'] = np.nan

    for num, name in enumerate(reg_list0, start=0):
        df_county['ICEV'] = np.where(df_county.region == name, temp['count'][num], df_county['ICEV'])

    for num, name in enumerate(reg_list1, start=len(reg_list0)):
        df_county['EV'] = np.where(df_county.region == name, temp['count'][num], df_county['EV'])

    df_county = df_county.fillna(0)

    df_county['EV_per'] = round(df_county['EV'] / (df_county['EV'] + df_county['ICEV']) * 100, 2)

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    df_county_str = df_county.copy()
    df_county_str.FIPS = df_county_str.FIPS.astype('str')
    df_county_str['FIPS'] = df_county_str['FIPS'].str.zfill(5)

    return df_county_str, counties


def map_plot(model_pred=None, ycolumn='y'):
    data = read_data(model_pred=model_pred)
    df_county = ca_county()
    df_county_str, counties = design_county(data, df_county, ycolumn=ycolumn)

    # plotting
    fig = px.choropleth_mapbox(df_county_str, geojson=counties, locations='FIPS', color='EV_per',
                               hover_name='region',
                               color_continuous_scale="Viridis",
                               #range_color=(0, math.ceil(df_county_str['EV_per'].max())),
                               range_color=(0, 100),
                               mapbox_style="carto-positron",
                               zoom=4.5,
                               center={"lat": 36.778259, "lon": -119.417931},
                               opacity=0.8,
                               labels={'EV_per': 'Percentage of EV'}
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
