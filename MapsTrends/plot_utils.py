import os
from matplotlib import cm, legend, pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from shapely.geometry import Point
from DataProcessing.grid_utils import grid_utils
from scipy.interpolate import griddata
import geopandas as gpd
import matplotlib.colors as mcolors
import seaborn as sns

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
    
    def plot_SOC(model_name, year_str, predictions, map_output_path):
        # Get predictions as a GeoDataFrame
        gdf = plot_utils.get_predictions_geoframe(predictions)
        soc = gdf['SOC']
        
        carbon_cmap = cm.viridis
        carbon_norm = mcolors.Normalize(vmin=soc.min(), vmax=soc.max())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(gdf['Lon'], gdf['Lat'], c=soc, cmap=carbon_cmap, norm=carbon_norm)
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        ax.set_title(f'{model_name} Predicted SOC (g/cm2) for South Africa in Year {year_str}', fontsize=16)

        # Plot South Africa shape boundaries
        grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Soil Organic Carbon (g/cm2)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Save the plot
        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path, bbox_inches='tight')
    
    def scatter_plot_predict_c_targetc(df, model_name, output_path):
        df = df.dropna(subset=['Target_C'])
        if df.empty:
            return

        # Plot the data
        plt.figure(figsize=(60, 20))
        plt.scatter(df['Target_C'], df['C'], color=sns.color_palette("viridis", as_cmap=True)(0.6), label='Data points')
        plt.plot([df['C'].min(), df['C'].max()], [df['C'].min(), df['C'].max()], color=sns.color_palette("viridis", as_cmap=True)(0.9), linestyle='--', label='Ideal fit')

        # Set equal scaling
        max_val = max(df['Target_C'].max(), df['C'].max())
            
        # Create bins of 0.5
        bin_width = 0.5
        min_bin = 0
        max_bin = np.ceil(max_val / bin_width) * bin_width

        plt.xlim(min_bin, max_bin)
        plt.ylim(min_bin, max_bin)

        # Set the ticks to match the bins
        plt.xticks(np.arange(min_bin, max_bin + bin_width, bin_width))
        plt.yticks(np.arange(min_bin, max_bin + bin_width, bin_width))
        
        plt.xlabel(f'Target Carbon (% by Mass)')
        plt.ylabel(f'Predicted Carbon (% by Mass)')
        plt.title(f'{model_name} Predicted Carbon vs Target Carbon')
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')

        #plt.show()
