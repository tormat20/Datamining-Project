import pandas as pd
import geopandas as gpd
import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mapclassify
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

#Generate csv data in lat and longitude
def get_csv_data():
    data = pd.read_csv('202406-capitalbikeshare-tripdata.csv')
    data = data.dropna()
    data['ended_at'] = pd.to_datetime(data['ended_at'])

    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['end_lng'], data['end_lat'], crs='EPSG:4326'))

    return data

#Generate csv data in meters
def get_csv_data_meter():
    data = pd.read_csv('202406-capitalbikeshare-tripdata.csv')
    data = data.dropna()
    data['ended_at'] = pd.to_datetime(data['ended_at'])

    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['end_lng'], data['end_lat'], crs='EPSG:4326'))
    data.to_crs(crs='EPSG:32634', inplace=True)

    return data

#Color for outliers
def my_colormap(value):
    if value == 0:
        return "green"
    return "red"

#Color for hotspots
def threshold_colormap(value):
    if value == 0:
        return "blue"
    elif value == 1:
        return "green"
    return "red"

# Input argument, df in meters 
# Returns outliers bool series 
def remove_outliers_if(df):
    points = df['geometry']
    df_coord = pd.DataFrame({'x': points.x, 'y': points.y})
    scaler = StandardScaler()
    df_coord_scaled = scaler.fit_transform(df_coord.to_numpy())
    iforest = IForest().fit(df_coord_scaled)
    outliers = iforest.predict(df_coord_scaled)
    return outliers

#To find the thresholds for low,medium and high density hotspots
def findThresholds(data,min,max):
    if data > ((2*((max-min)/3))+min):
        return 2
    elif data > ((max-min)/3)+min:
        return 1
    else:
        return 0