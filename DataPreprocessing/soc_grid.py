import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contextily as ctx
import re
from data_utils import data_utils

south_africa = data_utils.get_sa_shape()

def degrees_to_decimal(lat_or_lon):
    deg, minutes, seconds, direction =  re.split('[°\'"]', lat_or_lon)
    return (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['E', 'S'] else 1)

def get_conservation_SA_data():
    soc_2022_df = pd.read_excel(r'Data\FieldSamples\Mitsubishi_SOC_Baseline_December_2022.xlsx')['LAT', 'LON', 'C', 'SOILBD']
    soc_2022 = pd.DataFrame()
    soc_2022['source'] = 'Conservation South Africa'
    soc_2022['date'] = '12/01/2022'
    soc_2022['Lat'] = soc_2022_df['LAT'].apply(degrees_to_decimal)
    soc_2022['Lon'] = soc_2022_df['LON'].apply(degrees_to_decimal)
    soc_2022['C'] = soc_2022_df['C']
    soc_2022['BD'] = soc_2022_df['SOILBD']
    return soc_2022

def preprocess_data():
    soc_data = pd.read_excel(r'Data\FieldSamples\SOC Data from Heidi 20230124 - cleaned_additional.xlsx')
    soc_data = soc_data.append(get_conservation_SA_data(), ignore_index = True)
    soc_data['C_range'] = pd.cut(soc_data['C'], 
                                 bins=[-float('inf'), 0.5, 1, 2, 3, 4, float('inf')], 
                                 labels=["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"])
    soc_data['Date'] = pd.to_datetime(soc_data['date'])
    soc_data['Year'] = soc_data['date'].dt.year

    geometry = [Point(xy) for xy in zip(soc_data.Lon, soc_data.Lat)]
    soc_gdf = gpd.GeoDataFrame(soc_data, crs="EPSG:4326", geometry=geometry)
    soc_gdf.to_csv(r"DataPreprocessing\soc_gdf.csv", index=False)

def plot_map(data, title, color_col):
    color_mapping = {
    "<0.5": "red", "0.5-1": "orange", "1-2": "yellow",
    "2-3": "green", "3-4": "blue", ">4": "darkgreen"
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    south_africa.boundary.plot(ax=ax, linewidth=1)
    data.plot(column=color_col, ax=ax, legend=True, cmap=ListedColormap(color_mapping.values()))
    ctx.add_basemap(ax, crs=south_africa.crs.to_string())
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_C_distribution_by_year():
    soc_data = pd.read_csv(r'DataPreprocessing\soc_gdf.csv')
    for year in soc_data['Year'].unique():
        plot_map(soc_data[soc_data['Year'] == year], "Carbon % by Mass Distribution in South Africa for (1987-2018)", "C_range")
        print('Press any key to continue')

preprocess_data()
    

