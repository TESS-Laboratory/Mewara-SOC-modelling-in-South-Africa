import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class data_analysis:
    channel_names = {
                    'Landsat': ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI'], 
                    'Climate': ['Precipitation', 'Minimum Temp.', 'Maximum Temp.'],
                    'Terrain': ['DEM', 'Aspect', 'Slope', 'TWI']
                    }
    
    all_channels = ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI',
                'Precipitation', 'Minimum Temp.', 'Maximum Temp.',
                'DEM', 'Aspect', 'Slope', 'TWI']
    
    def get_carbon_range(carbon):
        if carbon <= 0.5:
            return "<= 0.5"
        elif carbon > 0.5 and carbon <= 1:
            return "0.5-1.0"
        elif carbon > 1 and carbon <= 2:
            return "1.0-2.0"
        elif carbon > 2 and carbon <= 3:
            return "2.0-3.0"
        elif carbon > 3 and carbon <= 4:
            return "3.0-4.0"
        else:
            return "> 4.0"

    
    def plot_pearson_coefficient(lat_lon_data, landsat_data, climate_data, terrain_data, targets):
        correlations = []

        df_lat_lon = pd.DataFrame(data=lat_lon_data, columns=['Lat', 'Lon'])
        df_targets = pd.DataFrame(data=targets, columns=['C'])
        df_landsat = data_analysis.populate_pearson_coefficient(input_data=landsat_data, input_data_type='Landsat', target_data=targets, correlations=correlations)
        df_climate = data_analysis.populate_pearson_coefficient(input_data=climate_data, input_data_type='Climate', target_data=targets, correlations=correlations)
        df_terrain = data_analysis.populate_pearson_coefficient(input_data=terrain_data, input_data_type='Terrain', target_data=targets, correlations=correlations)
        
        df = pd.concat([df_lat_lon, df_targets, df_landsat, df_climate, df_terrain], axis=1)
        df['Category'] = df['C'].apply(data_analysis.get_carbon_range)
        df['Category'] = df['Category'].astype(str)

        df.to_csv(f'DataProcessing/input_covariates.csv', index=False)

        data_analysis.heat_map(correlations=correlations, channels=data_analysis.all_channels)
        input('Press key to continue')

    def populate_pearson_coefficient(input_data_type, input_data, target_data, correlations):
        size = input_data.shape[-2]
        no_of_channels = input_data.shape[-1]
        df = pd.DataFrame()
        
        print(f"\nCalculating Pearson's Coefficient for {input_data_type}\n")

        for i in range(no_of_channels):
            channel_flat = input_data[:,:,:,i].flatten()
            mask = ~np.isnan(channel_flat)
            target_flat = np.repeat(target_data, size * size)
            corr, _ = pearsonr(channel_flat[mask], target_flat[mask])
            correlations.append(corr)
            df[data_analysis.channel_names[input_data_type][i]] = channel_flat
            print(f"Channel {i}: {corr}")
        
        return df

    def heat_map(correlations, channels):
        df = pd.DataFrame({'Channel': channels, 'Correlation': correlations})
        df_sorted = df.sort_values(by='Correlation', ascending=False)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        colors = ['blue' if val >= 0 else 'red' for val in df_sorted['Correlation']]
        plt.barh(df_sorted['Channel'], df_sorted['Correlation'], color=colors)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.title('Pearson Correlation Coefficients')
        plt.show()   



