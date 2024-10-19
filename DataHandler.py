import pandas as pd
import geopandas as gpd
import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mapclassify
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#--------------------------------------------------------------------
# Functions for data generation, either in lat and long or in meters
#--------------------------------------------------------------------

#Generate csv data in lat and longitude
def get_csv_data():
    data = pd.read_csv('202406-capitalbikeshare-tripdata.csv')

    #Data preprocessing (Removing missing and duplicate values)
    data = data.dropna() #Dropping all rows with missing values
    data = data[data.duplicated() == False] #Removing duplicates

    data['ended_at'] = pd.to_datetime(data['ended_at'])
    data['started_at'] = pd.to_datetime(data['started_at'])
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['end_lng'], data['end_lat'], crs='EPSG:4326'))

    return data

#Generate csv data in meters
def get_csv_data_meter():
    data = pd.read_csv('202406-capitalbikeshare-tripdata.csv')

    #Data preprocessing (Removing missing and duplicate values)
    data = data.dropna() #Dropping all rows with missing values
    data = data[data.duplicated() == False] #Removing duplicates

    data['ended_at'] = pd.to_datetime(data['ended_at'])
    data['started_at'] = pd.to_datetime(data['started_at'])
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['end_lng'], data['end_lat'], crs='EPSG:4326'))
    data.to_crs(crs='EPSG:32634', inplace=True)

    return data

#----------------------------
# Functions color generation
#----------------------------

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

#-----------------------------------------
# Function for outlier handling using IF
#-----------------------------------------

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

#-------------------------------------------------------
# Function for creating three threshholds for hotspots
#-------------------------------------------------------

#To find the thresholds for low,medium and high density hotspots
def findThresholds(data,min,max):
    if data > ((2*((max-min)/3))+min):
        return 2
    elif data > ((max-min)/3)+min:
        return 1
    else:
        return 0
    
#--------------------------
# Functions for clustering
#--------------------------
    
# Input argument, df in meters 
# Returns outliers bool series 
def clustering_DBScan(df):
    points = df['geometry']
    df_coord = pd.DataFrame({'x': points.x, 'y': points.y})
    scaler = StandardScaler()
    df_coord_scaled = scaler.fit_transform(df_coord.to_numpy())

    def create_model(eps, min_samples, df):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
        return dbscan.labels_

    def search_params(df):
        params = {
            's': -10,
            'eps': 0,
            'min_samples': 0
        }

        for eps in [0.001, 0.05, 0.01, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]:
            for min_samples in [5, 20, 25, 30, 50, 70, 90, 130, 160, 190, 220, 250, 280, 310]:
                labels = create_model(eps,min_samples,df.copy())

                #Cant calculate silhouette score with 1 cluster
                if len(np.unique(labels)) == 1:
                     continue
                avg_silhouette = silhouette_score(df,labels=labels)

                if avg_silhouette > params['s']:
                    params['s'] = avg_silhouette
                    params['eps'] = eps
                    params['min_samples'] = min_samples 
        return params

    opt_params = search_params(df_coord_scaled)
    return  create_model(opt_params['eps'],opt_params['min_samples'],df_coord_scaled),opt_params

# Input argument, df in meters 
# Returns outliers bool series 
def clustering_Kmeans(df):
    points = df['geometry']
    df_coord = pd.DataFrame({'x': points.x, 'y': points.y})
    scaler = StandardScaler()
    df_coord_scaled = scaler.fit_transform(df_coord.to_numpy())

    def create_model(k, df):
        dbscan = KMeans(n_clusters=k).fit(df)
        return dbscan.labels_

    def search_params(df):
        params = {
            's': -10,
            'k': 0
        }

        for k in range(1,100):
            labels = create_model(k,df.copy())

            #Cant calculate silhouette score with 1 cluster
            if len(np.unique(labels)) == 1:
                    continue
            avg_silhouette = silhouette_score(df,labels=labels)

            if avg_silhouette > params['s']:
                params['s'] = avg_silhouette
                params['k'] = k
        return params

    opt_params = search_params(df_coord_scaled)
    return  create_model(opt_params['k'],df_coord_scaled),opt_params

#--------------------------------------
# Functions for the anomaly detection
#--------------------------------------

#Convert the string values of membership type and ride type to numbers for anomaly detection
def convertToNumber(data,uniques):
    for index in range(len(uniques)):
        if data == uniques[index]:
            return index
    return -1

#Extracts and converts features to values that can be used to perform anomaly detection
def dataTransformation(df):
    df["rideable_type"] = df["rideable_type"].apply(convertToNumber,uniques=(df["rideable_type"].unique()))
    df["member_casual"] = df["member_casual"].apply(convertToNumber,uniques=(df["member_casual"].unique()))

    df.to_crs(crs='EPSG:32634', inplace=True) #Converting the values to meters if a lon/lat DF was provided
    end_points = df['geometry']

    end_coords = pd.DataFrame({'end_x': end_points.x, 'end_y': end_points.y})
    scaler = StandardScaler()
    end_coords_scaled = scaler.fit_transform(end_coords)

    df.reset_index(inplace=True) #To allow the scaled values to be added
    df['end_x'] = pd.Series(end_coords_scaled[:,0])
    df['end_y'] = pd.Series(end_coords_scaled[:,1])

    #Adding and converting the starting points aswell
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start_lng'], df['start_lat'], crs='EPSG:4326'))
    df.to_crs(crs='EPSG:32634', inplace=True)
    df.drop(columns='index',inplace=True)

    start_points = df['geometry']

    start_coords = pd.DataFrame({'start_x': start_points.x, 'start_y': start_points.y})
    start_coords_scaled = scaler.fit_transform(start_coords)

    df['start_x'] = pd.Series(start_coords_scaled[:,0])
    df['start_y'] = pd.Series(start_coords_scaled[:,1])

    return df

#Finding outliers based on membership class, ride type, start position (x and y) and end position
def find_outliers_if(df):
    #Transforming the before finding outliers, to utilize as much information as possible
    transformed_df = dataTransformation(df)

    df_new = transformed_df.drop(columns=["ride_id","started_at","ended_at","start_station_name","start_station_id","end_station_name","end_station_id","start_lat","start_lng","end_lat","end_lng","geometry"])
    df_new = df_new.to_numpy()

    iforest = IForest(contamination=0.02).fit(df_new) #Assuming that the contamination is about 2%. The seemingly arbitrary choice of 2% was chosen based on the result from the PCA plots for the outliers
    outliers = iforest.predict(df_new)
    return outliers

def calc_distance():
    return 0

#Finding the distance between start and end point
def find_distance(df):
    #Take out the end positions
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['end_lng'], df['end_lat'], crs='EPSG:4326'))
    df.to_crs(crs='EPSG:32634', inplace=True) #Converting the values to meters if a lon/lat DF was provided

    #Storing the positions (unscaled) seperatly
    end_points = df['geometry']
    coords = pd.DataFrame({'end_x': end_points.x, 'end_y': end_points.y})

    #Adding and converting the starting points aswell
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start_lng'], df['start_lat'], crs='EPSG:4326'))
    df.to_crs(crs='EPSG:32634', inplace=True)
    start_points = df['geometry']

    #Storing the positions (unscaled) seperatly
    coords['start_x'] = start_points.x
    coords['start_y'] = start_points.y
    coords = coords.to_numpy()

    #Returning the travel distances
    return np.sqrt(np.square(coords[:,0]-coords[:,2])+np.square(coords[:,1]-coords[:,3]))