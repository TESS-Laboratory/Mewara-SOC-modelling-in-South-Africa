import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import re

#south_africa = data_utils.get_sa_shape()
south_africa = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).query('name == "South Africa"')

def get_soc_data():
    return pd.read_csv(r'DataPreprocessing/soc_gdf.csv')

def degrees_to_decimal(lat_or_lon):
    # Skip conversion if the input is already a decimal number
    try:
        return float(lat_or_lon)
    except ValueError:
        pass

    # Normalize the input to replace curly quotes with straight quotes
    lat_or_lon = lat_or_lon.strip().replace('’', "'").replace('“', '"').replace('”', '"').replace("''", '"')
    
    # Split the input string and discard empty strings
    parts = re.split('[°\'"]', lat_or_lon)
    parts = [p for p in parts if p]
    deg, minutes, seconds = 0, 0, 0
    direction = ''
    
    if len(parts) == 1:
        deg = parts[0]
    elif len(parts) == 4:
        deg, minutes, seconds, direction = parts
    else:
        raise ValueError("Input format must be 'degrees°minutes'seconds direction")

    # Calculate the decimal value
    decimal = float(deg) + float(minutes)/60 + float(seconds)/(60*60)
    
    # Apply the direction
    if direction.strip() in ['S', 'W']:
        decimal *= -1
    
    return decimal

def get_conservation_SA_data():
    soc_2022_df = pd.read_excel(r'Data/FieldSamples/Mitsubishi_SOC_Baseline_December_2022.xlsx')

    # Apply degrees_to_decimal only if the value is not empty
    soc_2022_df['Lat'] = soc_2022_df['LAT'].apply(lambda x: degrees_to_decimal(x) if pd.notnull(x) and x != '' else None)
    soc_2022_df['Lon'] = soc_2022_df['LONG'].apply(lambda x: degrees_to_decimal(x) if pd.notnull(x) and x != '' else None)
    
    soc_2022 = pd.DataFrame({
        'Source': 'Conservation South Africa',
        'Date': '12/01/2022',
        'Lat': soc_2022_df['Lat'],
        'Lon': soc_2022_df['Lon'],
        'C': soc_2022_df['C'],
        'BD': soc_2022_df['SOILBD'],
        'UniqueID': soc_2022_df['Monsterverwysing / Sample Reference']
    })
    
    # Drop rows with empty latitude or longitude
    soc_2022 = soc_2022.dropna(subset=['Lat', 'Lon'])

    return soc_2022

def preprocess_data():
    # Read SOC data from Excel file
    soc_data = pd.read_excel(r'Data/FieldSamples/SOC Data from Heidi 20230124 - cleaned_additional.xlsx')

    # Concatenate with conservation data for South Africa if available
    soc_data = pd.concat([soc_data, get_conservation_SA_data()], ignore_index=True)

    # Bin C values into categories
    soc_data['C_range'] = pd.cut(soc_data['C'], 
                                 bins=[-float("inf"), 0.5, 1, 2, 3, 4, float("inf")], 
                                 labels=["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"])

    # Convert 'Date' column to datetime format and extract year and month
    soc_data['Date'] = pd.to_datetime(soc_data['Date'])
    soc_data['Year'] = soc_data['Date'].dt.year
    soc_data['Month'] = soc_data['Date'].dt.month

    soc_data = soc_data.dropna(subset=['Lat', 'Lon']) # Drop rows with empty latitude or longitude
    soc_data = soc_data[soc_data['C'] <= 20] # Drop rows where C % is greater 20
    soc_data.drop_duplicates(['Source', 'Date', 'Lat', 'Lon', 'C', 'BD'], inplace=True) # Drop duplicates

    geometry = [Point(xy) for xy in zip(soc_data.Lon, soc_data.Lat)]
    soc_gdf = gpd.GeoDataFrame(soc_data, crs="EPSG:4326", geometry=geometry)
    soc_gdf.to_csv(r"DataProcessing/soc_gdf.csv", index=False)

#preprocess_data()