import os
from matplotlib import legend, pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from shapely.geometry import Point
from DataProcessing.grid_utils import grid_utils
from scipy.interpolate import griddata
import geopandas as gpd
import matplotlib.colors as mcolors

class plot_utils:
    def get_predictions_geoframe(predictions):
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        gdf = grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid, soil_data=predictions)

        return gdf
    
    def get_carbon_mapping():
        carbon_mapping = grid_utils.get_carbon_mapping()
        # Create boundaries and colors for the colormap
        boundaries = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        colors = [carbon_mapping["<0.5"], carbon_mapping["0.5-1.0"], carbon_mapping["1.0-1.5"],
                carbon_mapping["1.5-2.0"], carbon_mapping["2.0-2.5"], carbon_mapping["2.5-3.0"], 
                carbon_mapping["3.0-3.5"], carbon_mapping[">3.5"]]
        
        # Create the colormap and norm
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        
        return cmap, norm
    
    def plot_predictions(year, predictions, map_output_path):
        gdf = plot_utils.get_predictions_geoframe(predictions)

        carbon_cmap, carbon_norm = plot_utils.get_carbon_mapping()
        
        # Plot the result
        plt.figure(figsize=(10, 8))
        plt.scatter(gdf['Lon'], gdf['Lat'], c=gdf['C'], cmap=carbon_cmap, norm=carbon_norm)
        plt.xlabel('Longitude', fontsize=16)
        plt.ylabel('Latitude', fontsize=16)
        plt.title(f'Predicted Carbon (% by Mass) Distribution in South Aftrica in Year {year}', fontsize=16)

        # Create custom legend
        handles = [Patch(color=color, label=label) for label, color in grid_utils.get_carbon_mapping().items()]
        plt.legend(handles=handles, title=r'Carbon (% by mass)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title_fontsize=16)

        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path, bbox_inches='tight')

        #plt.show()
   
    def scatter_plot_predict_c_targetc(df, output_path):
        df = df.dropna(subset=['Target_C'])

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(df['C'], df['Target_C'], color='blue', label='Data points')
        plt.plot([df['C'].min(), df['C'].max()], [df['C'].min(), df['C'].max()], color='red', linestyle='--', label='Ideal fit')

        plt.xlabel(f'Predicted C (% by Mass)')
        plt.ylabel(f'Target C (% by Mass)')
        plt.title(f'Predicted Carbon vs Target Carbon')
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')

        #plt.show()

pred_1999 = pd.read_csv('Maps\Best_RF_Model\Predictions\predictions_1999.csv')
pred_2008 = pd.read_csv('Maps\Best_RF_Model\Predictions\predictions_2008.csv')
pred_2018 = pd.read_csv('Maps\Best_RF_Model\Predictions\predictions_2018.csv')

combined_df = pd.concat([pred_1999, pred_2008, pred_2018], ignore_index=True)

#plot_utils.scatter_plot_predict_c_targetc(combined_df, r'Maps\Best_RF_Model\ScatterPlots\rf_scatter_plot.png')

#plot_utils.plot_predictions(year=2008, predictions=pred_2008, map_output_path=r'Maps\Best_RF_Model\PredictedMaps\predictions_2008.png')
