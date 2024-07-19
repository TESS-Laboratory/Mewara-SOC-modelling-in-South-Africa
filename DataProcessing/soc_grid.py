import pandas as pd
import re
from DataProcessing import grid_utils
from Model.training_data_utils import training_data_utils

def calculate_soc(soc_data):
    soc_data['SOC'] = soc_data['C'] * soc_data['BulkDensity'] * 20
    return soc_data

def save_hex_grid_with_bulk_density():
    hex_grid = grid_utils.grid_utils.get_hex_grid(0.1)
    lat_lon_pairs = list(set(zip(hex_grid['Hex_Center_Lat'], hex_grid['Hex_Center_Lon'])))
    patches = training_data_utils.get_bd_patches_dict(lat_lon_pairs=lat_lon_pairs)
    for lat, lon in lat_lon_pairs:
        hex_grid.loc[(hex_grid['Hex_Center_Lat'] == lat) & (hex_grid['Hex_Center_Lon'] == lon), 'Hex_Center_BD'] = patches.get((lat, lon))

    hex_grid.to_csv(r'DataProcessing/hex_grid.csv', index=False)

def merge_bulk_density_isda(soc_data):
    soc_empty_bulk_density = soc_data[soc_data['BD'].isna()]
    lat_lon_pairs = list(set(zip(soc_empty_bulk_density['Lat'], soc_empty_bulk_density['Lon'])))
    patches = training_data_utils.get_bd_patches_dict(lat_lon_pairs=lat_lon_pairs)
    for lat, lon in lat_lon_pairs:
        soc_data.loc[(soc_data['Lat'] == lat) & (soc_data['Lon'] == lon), 'BD_iSDA'] = patches.get((lat, lon))

    soc_data['BulkDensity'] = soc_data['BD'].fillna(soc_data['BD_iSDA'])
    return soc_data

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

def get_iSDA_data():
    isda_2018_df = pd.read_csv(r'Data/FieldSamples/iSDA.csv')
    isda_2018 = pd.DataFrame({
        'Source': 'iSDA',
        'Date': isda_2018_df['Date'],
        'Lat': isda_2018_df['Lat'],
        'Lon': isda_2018_df['Lon'],
        'C': isda_2018_df['C'],
        'UniqueID': isda_2018_df['UniqueID']
    })

    return isda_2018

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
    soc_data = pd.concat([soc_data, get_conservation_SA_data(), get_iSDA_data()], ignore_index=True)

    # Convert 'Date' column to datetime format and extract year and month
    soc_data['Date'] = pd.to_datetime(soc_data['Date'])
    soc_data['Year'] = soc_data['Date'].dt.year
    soc_data['Month'] = soc_data['Date'].dt.month

    soc_data = soc_data.dropna(subset=['Lat', 'Lon']) # Drop rows with empty latitude or longitude
    soc_data = soc_data[(soc_data['C'] <= 60)] # Drop rows where C % is greater 60
    soc_data.drop_duplicates(['Source', 'Date', 'Lat', 'Lon', 'C', 'BD'], inplace=True) # Drop duplicates

    soc_data = merge_bulk_density_isda(soc_data=soc_data)
    soc_data = calculate_soc(soc_data=soc_data)

    save_soc_hex_grid(soc_data)

def save_soc_hex_grid(soil_data):
    hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
    soc_hex_grid = grid_utils.grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid,
                                            soil_data=soil_data) 
    soc_hex_grid.to_csv(r'DataProcessing/soc_hex_grid.csv', index=False)
    
    soc_hex_avg_c = grid_utils.grid_utils.get_avg_c_each_grid(df=soc_hex_grid,
                                                    group_by_cols=['Hex_ID'],
                                                    avg_col='C',
                                                    avgc_col_rename='Avg_C')
    
    hex_grid_avg_c = pd.merge(hex_grid, soc_hex_avg_c, on='Hex_ID', how='left')
    
    grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=hex_grid_avg_c, 
                            title=f'Long Term Average Carbon (% by Mass) of South Africa (1986-2022)',
                            geometry_col='geometry',
                            savePlot=True,
                            output_plot_path='Plots/C_HeatMap.png')

#save_hex_grid_with_bulk_density()
#preprocess_data()
