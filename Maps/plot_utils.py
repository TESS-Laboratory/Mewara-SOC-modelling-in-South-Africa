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
        boundaries = [-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, np.inf]
        colors = [carbon_mapping["<0.5"], carbon_mapping["0.5-1.0"], carbon_mapping["1.0-1.5"],
                carbon_mapping["1.5-2.0"], carbon_mapping["2.0-2.5"], carbon_mapping["2.5-3.0"], 
                carbon_mapping["3.0-3.5"], carbon_mapping["3.5-4.0"], carbon_mapping[">4.0"]]
        
        # Create the colormap and norm
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        
        return cmap, norm
    
    def plot_predictions(model_name, year_str, predictions, map_output_path, carbon_col='C'):
        gdf = plot_utils.get_predictions_geoframe(predictions)

        carbon_cmap, carbon_norm = grid_utils.get_carbon_mapping_bins_colors()
        
        # Plot the result
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(gdf['Lon'], gdf['Lat'], c=gdf[carbon_col], cmap=carbon_cmap, norm=carbon_norm)
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        ax.set_title(f'{model_name} Predicted Carbon (% by Mass) for South Africa in Year {year_str}', fontsize=16)

        grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')

        # Create custom legend
        handles = [Patch(color=color, label=label) for label, color in grid_utils.get_carbon_mapping().items()]
        legend = ax.legend(handles=handles, title=r'Carbon (% by mass)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15, title_fontsize=16)

        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path, bbox_inches='tight', bbox_extra_artists=[legend])

        #plt.show()
   
    def scatter_plot_predict_c_targetc(df, model_name, output_path):
        df = df.dropna(subset=['Target_C'])

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Target_C'], df['C'], color='blue', label='Data points')
        plt.plot([df['C'].min(), df['C'].max()], [df['C'].min(), df['C'].max()], color='red', linestyle='--', label='Ideal fit')

        plt.ylabel(f'Predicted C (% by Mass)')
        plt.xlabel(f'Target C (% by Mass)')
        plt.title(f'{model_name} Predicted Carbon vs Target Carbon')
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')

        #plt.show()
