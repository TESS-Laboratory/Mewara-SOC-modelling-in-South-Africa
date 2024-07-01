from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from DataProcessing.grid_utils import grid_utils
from Maps.test_metrics import test_metrics
import DataProcessing.grid_utils
from scipy.interpolate import griddata

class map_utils:
    @staticmethod
    def create_map(year, start_month, end_month, model, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, skip_predictions=False):
        tm = test_metrics(model=model)
        output_dir = f'Maps/{model.get_model_name()}'
        error_output_path = f'{output_dir}/Errors/error_{year}.txt'
        predictions_output_path = f'{output_dir}/Predictions/predictions_{year}.csv'
        predicted_plot_output_path = f'{output_dir}/PredictedMaps/predicted_{year}.png'

        avg_cols = ['Hex_ID', 'Year']

        hex_grid_avg_c = map_utils.get_hex_grid_avg_c(year)
        
        map_utils.plot_actual_map(year=year, hex_grid=hex_grid_avg_c)

        if not skip_predictions:
            print(f'\nFetching predictions for year {year}\n')
            predictions = tm.predict(year=year,
                    start_month=start_month,
                    end_month=end_month,
                    hex_grid_year_month=hex_grid_avg_c,
                    patch_size_meters_landsat=patch_size_meters_landsat,
                    patch_size_meters_climate=patch_size_meters_climate, patch_size_meters_terrain=patch_size_meters_terrain,
                    save=True,
                    output_path=predictions_output_path,
                    error_output_path=error_output_path)
            
        map_utils.plot_predicted_map(year=year,
                                     predictions=predictions,
                                     predictions_plot_path=predicted_plot_output_path)

    @staticmethod
    def plot_actual_map(year, hex_grid):
        DataProcessing.grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=hex_grid,       
                                                            title=f'Average C (% by Mass) for South Africa in Year {year}',
                                                            geometry_col='geometry_x',
                                                            savePlot=True,
                                                            output_plot_path=f'Maps/ActualMaps/Actual_{year}.png')
        #input('press key to continue')

    @staticmethod
    def get_hex_grid_avg_c(year):
        soc_hex_grid = pd.read_csv(r'DataProcessing/soc_hex_grid.csv')
        soc_hex_grid = soc_hex_grid[soc_hex_grid['Year'] == year]

        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        hex_grid['Year'] = year
        months_df = pd.DataFrame({'Month': range(1, 13)})
        hex_grid = hex_grid.merge(months_df, how='cross')

        hex_grid_soc_hex = pd.merge(hex_grid, soc_hex_grid, on=['Hex_ID', 'Year', 'Month'], how='left')

        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_avg_c_each_grid(df=hex_grid_soc_hex,
                                                        group_by_cols=['Hex_ID', 'Hex_Center_Lat_x', 'Hex_Center_Lon_x', 'Year', 'Month', 'geometry_x'],
                                                        avg_col='C',
                                                        avgc_col_rename='Avg_C')
        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_geoframe(soc_hex_avg_c, 'geometry_x')
                                         
        return soc_hex_avg_c
        
    @staticmethod
    def plot_predicted_map(year, predictions, predictions_plot_path):
        if predictions is None or len(predictions) == 0:
            print(f'\nPredictions is missing for year {year}\n')
            return None

        predictions = predictions[predictions['Year']== year]
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        pred_hex_df = DataProcessing.grid_utils.grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid,
                                             soil_data=predictions) 

        pred_grid_avg_c_year = DataProcessing.grid_utils.grid_utils.get_avg_c_each_grid(df=pred_hex_df,
                                                                   group_by_cols=['Hex_ID', 'geometry', 'Year'],
                                                                   avg_col='C',
                                                                   avgc_col_rename='Avg_C')
        
        DataProcessing.grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=pred_grid_avg_c_year,       
                                                            geometry_col='geometry',
                                                            title=f'Predicted C (% by Mass) for South Africa in Year {year}',
                                                            savePlot=True,
                                                            output_plot_path=predictions_plot_path)
        #input('press key to continue')


    def plot_interpolated_map(predictions_path):
        predictions = pd.read_csv(predictions_path)
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        gdf = DataProcessing.grid_utils.grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid,
                                             soil_data=predictions) 

        # Extract coordinates and SOC values
        points = gdf[['Hex_Center_Lat', 'Hex_Center_Lon']].values
        values = gdf['C'].values

        gdf = gdf.dropna(subset=['C'])

        # Define the grid over which to interpolate
        grid_lat, grid_lon = np.mgrid[
            gdf['Hex_Center_Lat'].min():gdf['Hex_Center_Lat'].max():100j,
            gdf['Hex_Center_Lon'].min():gdf['Hex_Center_Lon'].max():100j
        ]

        # Perform the interpolation
        grid_C = griddata(points, values, (grid_lat, grid_lon), method='cubic')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        DataProcessing.grid_utils.grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        #gdf.plot(column='C', ax=ax, legend=True, cmap='viridis', edgecolor='black')

        # Plot the interpolated SOC map
        plt.imshow(grid_C.T, extent=(gdf['Hex_Center_Lon'].min(), gdf['Hex_Center_Lon'].max(), gdf['Hex_Center_Lat'].min(), gdf['Hex_Center_Lat'].max()), origin='lower', cmap='inferno', alpha=0.6)

        plt.title('Interpolated SOC Content Map of South Africa')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='SOC Content')
        plt.show()
        input('press')

#map_utils.plot_interpolated_map(r'Maps\PredictedMaps\RF_LTrue_CTrue_TTrue_30\Predictions\predictions_2008.csv')