import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode
from urllib.request import urlopen
import json
import plotly.express as px
init_notebook_mode(connected=True)


def read_data(model_pred=None):
    data = pd.read_csv('../output_files/data.csv')
    X_test = pd.read_csv('../output_files/X_test.csv')
    y_test = pd.read_csv('../output_files/y_test.csv')

    data = data[data.index.isin(X_test.index)]
    data.drop(data[data['region'] == 7].index, inplace=True)  # drop "I don't know"

    # replace number with name
    region_names = ['Central', 'LA', 'Rest', 'SD', 'SF', 'Sac']
    for num, name in enumerate(region_names, start=1):
        data.region = data.region.replace(num, name)

    if model_pred == 'rf':
        pred_rf = pd.read_csv('../output_files/y_pred_rf.csv')
        data['pred_rf'] = pred_rf

    if model_pred == 'logit':
        pred_log = pd.read_csv('../output_files/y_pred_log.csv')
        data['pred_rf'] = pred_log

    return data, X_test, y_test


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
    :param ycolumn: 'y' or 'pred_rf' or 'pred_log'
    :return:
    """
    region_names = ['Central', 'LA', 'Rest', 'SD', 'SF', 'Sac']

    temp = data.groupby(ycolumn)['region'].value_counts().sort_index()
    df_county['ICEV'] = np.nan
    df_county['EV'] = np.nan
    df_county['EV_per'] = np.nan

    for num, name in enumerate(region_names):
        df_county['ICEV'] = np.where(df_county.region == name, temp[0][num], df_county['ICEV'])
        df_county['EV'] = np.where(df_county.region == name, temp[1][num], df_county['EV'])
        df_county['EV_per'] = np.where(df_county.region == name,
                                       round(temp[1][num] / (temp[0][num] + temp[1][num]) * 100, 2),
                                       df_county['EV_per'])

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    df_county_str = df_county.copy()
    df_county_str.FIPS = df_county_str.FIPS.astype('str')
    df_county_str['FIPS'] = df_county_str['FIPS'].str.zfill(5)

    return df_county_str, counties


def map_plot(df_county_str, counties):
    fig = px.choropleth_mapbox(df_county_str, geojson=counties, locations='FIPS', color='EV_per',
                               hover_name='region',
                               color_continuous_scale="Viridis",
                               range_color=(0, 16),
                               mapbox_style="carto-positron",
                               zoom=4.5,
                               center={"lat": 36.778259, "lon": -119.417931},
                               opacity=0.8,
                               labels={'EV_per': 'Percentage of EV'}
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
