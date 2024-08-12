import os
from matplotlib import cm, legend, pyplot as plt
from matplotlib.patches import Patch
from scipy.linalg import svd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from DataProcessing.grid_utils import grid_utils
from scipy.interpolate import griddata
import geopandas as gpd
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde

class plot_utils:
    def get_predictions_geoframe(predictions):
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        gdf = grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid, soil_data=predictions)
        return gdf
    
    def get_carbon_mapping():
        carbon_mapping = grid_utils.get_carbon_mapping()
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
        soc = gdf['SOC'] * 10
        
        carbon_cmap = cm.viridis
        carbon_norm = mcolors.Normalize(vmin=soc.min(), vmax=soc.max())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(gdf['Lon'], gdf['Lat'], c=soc, cmap=carbon_cmap, norm=carbon_norm)
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        ax.set_title(f'{model_name} Predicted SOC Stock (kg/m2) for South Africa in Year {year_str}', fontsize=16)

        # Plot South Africa shape boundaries
        grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Soil Organic Carbon Stock (kg/m2)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Save the plot
        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path, bbox_inches='tight')
    
    def total_least_squares(X, Y):
        # Formulate matrix
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        A = np.hstack((X, Y))
        
        # Perform SVD
        U, S, Vt = svd(A)
        
        # TLS solution is the last column of V (V transposed is Vt, so last row of Vt)
        v = Vt[-1, :]
        
        # Calculate slope and intercept
        slope = -v[0] / v[1]
        intercept = np.mean(Y) - slope * np.mean(X)
        
        return slope, intercept

    def scatter_plot_predict_c_targetc(df, model_name, output_path):
        df = df.dropna(subset=['Target_C'])
        if df.empty:
            return

        # Extract target and predicted values
        X = df['C'].values
        Y = df['Target_C'].values
        
        # Fit a total least squares line
        slope, intercept = plot_utils.total_least_squares(X, Y)
        
        # Plot the data
        plt.figure(figsize=(12, 12))
        plt.scatter(df['Target_C'], df['C'], color=sns.color_palette("viridis", as_cmap=True)(0.6), label='Data points')
        
        # Plot the TLS line
        x_vals = np.linspace(df['Target_C'].min(), df['Target_C'].max(), 20)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color=sns.color_palette("viridis", as_cmap=True)(0.9), linestyle='-', label='TLS Fit')

        # Set equal scaling
        min_val = min(df['Target_C'].min(), df['C'].min())
        max_val = 10 # max(df['Target_C'].max(), df['C'].max())
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # Create bins of 0.5
        bin_width = 0.5
        min_bin = np.floor(min_val / bin_width) * bin_width
        max_bin = np.ceil(max_val / bin_width) * bin_width

        # Set the ticks to match the bins
        plt.xticks(np.arange(min_bin, max_bin + bin_width, bin_width), fontsize=12)
        plt.yticks(np.arange(min_bin, max_bin + bin_width, bin_width), fontsize=12)
        
        plt.xlabel('Target Carbon (% by Mass)', fontsize=16)
        plt.ylabel('Predicted Carbon (% by Mass)', fontsize=16)
        plt.title(f'{model_name} Predicted Carbon vs Target Carbon', fontsize=16)
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')

        #plt.show()

    def density_plot_predict_c_targetc(df, model_name, output_path):
        plt.figure(figsize=(10, 8))
        
        sns.kdeplot(df['Target_C'], bw_adjust=0.5, fill=True, label='Predicted SOC Density', color='blue')
        sns.kdeplot(df['C'], bw_adjust=0.5, fill=True, label='Target SOC Density', color='orange')

        plt.xlabel('Soil Organic Carbon (% by Mass)', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.title(f'{model_name} Soil Organic Carbon Density', fontsize=16)
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')

    def plot_Biome_DensityPlot(biome_trends, biome_trends_col, map_output_path):
        soc = biome_trends[biome_trends_col]
        biomes = biome_trends['Biome']
        
        unique_biomes = sorted(biomes.unique())
        carbon_cmap = cm.viridis
        carbon_norm = mcolors.Normalize(vmin=soc.min(), vmax=soc.max())

        fig, ax = plt.subplots(figsize=(10, 8))

        # Set y-ticks and labels for the biomes
        ax.set_yticks(np.arange(len(unique_biomes)))
        ax.set_yticklabels(unique_biomes, fontsize=16)
        
        density_scaling_factor = 1.5

        for idx, biome in enumerate(unique_biomes):
            subset = biome_trends[biome_trends['Biome'] == biome]
            soc_values = subset[biome_trends_col].dropna()
            
            if not soc_values.empty:
                kde = gaussian_kde(soc_values, bw_method=0.5)
                soc_range = np.linspace(soc_values.min(), soc_values.max(), 1000)
                density = kde(soc_range) * density_scaling_factor
                for i in range(len(soc_range) - 1):
                    x0, x1 = soc_range[i], soc_range[i + 1]
                    y0, y1 = density[i], density[i + 1]
                    
                    # Calculate the color for this segment
                    color = carbon_cmap(carbon_norm((x0 + x1) / 2))
                    
                    # Fill between the baseline (idx) and the density curve
                    ax.fill_between([x0, x1], idx, [idx + y0, idx + y1], color=color)
                    ax.plot([x0, x1], [idx + y0, idx + y1], color=color)

        ax.set_xlabel('Soil Organic Carbon Stock (kg/m2)', fontsize=16)
        #ax.set_title(f'SOC Distribution by Biome', fontsize=16)

        # Save the plot
        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path, bbox_inches='tight')
        plt.close()

    def plot_Biome_Trends(biome_trends, biome_trends_col, map_output_path):
        # Get predictions as a GeoDataFrame
        soc = biome_trends[biome_trends_col]
        
        carbon_cmap = cm.viridis
        carbon_norm = mcolors.Normalize(vmin=soc.min(), vmax=soc.max())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(biome_trends['Lon'], biome_trends['Lat'], c=soc, cmap=carbon_cmap, norm=carbon_norm)
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        #ax.set_title(f'Predicted SOC Stock (g/cm2) for South Africa', fontsize=16)

        # Plot South Africa shape boundaries
        grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Soil Organic Carbon Stock (kg/m2)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Save the plot
        os.makedirs(os.path.dirname(map_output_path), exist_ok=True)
        plt.savefig(map_output_path)

#plot_utils.plot_Biome_Trends(biome_trends=pd.read_csv('MapsTrends/RF_Model/Trends/Biome_Trends.csv'), biome_trends_col='Mean_SOC', map_output_path='MapsTrends/RF_Model/Trends/Biome_SOC.png')
#plot_utils.plot_Biome_DensityPlot(biome_trends=pd.read_csv('MapsTrends/RF_Model/Trends/Biome_Trends.csv'), biome_trends_col='Mean_SOC', map_output_path='MapsTrends/RF_Model/Trends/Biome_SOC_Density.png')
