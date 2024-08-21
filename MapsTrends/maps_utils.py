import os
import numpy as np
import pandas as pd
from DataProcessing import grid_utils
from MapsTrends import plot_utils, test_metrics, trends_analysis

class map_utils:
    @staticmethod
    def create_map(year, model, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, landsat_bands, climate_bands, terrain_bands, skip_predictions=False):
        tm = test_metrics.test_metrics(model=model)
        model_name = model.get_model_name()
        output_dir = f'MapsTrends/{model_name}_{patch_size_meters_landsat}'.replace('.keras', '')
        error_output_path = f'{output_dir}/Errors/error_{year}.txt'
        predictions_folder = f'{output_dir}/Predictions'
        predictions_output_path = f'{predictions_folder}/predictions_{year}.csv'
        predicted_plot_output_path = f'{predictions_folder}/PredictedMaps/soc_{year}.png'
        predicted_c_output_path = f'{predictions_folder}/PredictedCarbonMaps/c_{year}.png'

        soc_mean_targets = map_utils.get_hex_grid_mean_C_SOC(year)
        
        if not skip_predictions:
            print(f'\nFetching predictions for year {year}\n')
            predictions = tm.predict(year=year,
                    hex_grid_year=soc_mean_targets,
                    start_month=start_month,
                    end_month=end_month,
                    patch_size_meters_landsat=patch_size_meters_landsat,
                    patch_size_meters_climate=patch_size_meters_climate, patch_size_meters_terrain=patch_size_meters_terrain,
                    landsat_bands=landsat_bands,
                    climate_bands=climate_bands,
                    terrain_bands=terrain_bands,
                    save=True,
                    output_path=predictions_output_path,
                    error_output_path=error_output_path)
         
        plot_utils.plot_utils.plot_SOC(model_name=model.__class__.__name__, 
                                       year_str=year,
                                       predictions=predictions,
                                       map_output_path=predicted_plot_output_path)
        
        map_utils.plot_predicted_points(year=year, 
                                        predictions=predictions, 
                                        predictions_plot_path=predicted_c_output_path)
    
        map_utils.plot_combined_predictions_scatter_density_plots(model_name=model.__class__.__name__, predictions_folder=predictions_folder)

        trends_analysis.trends_analysis.save_biome_trends(predictions_folder=predictions_folder, output_folder=f'{output_dir}/Trends')

    @staticmethod
    def plot_actual_map(year, hex_grid_avg_c):
        grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=hex_grid_avg_c,       
                                            title=f'Average Carbon (% by Mass) for South Africa in Year {year}',
                                            geometry_col='geometry_x',
                                            savePlot=True,
                                            output_plot_path=f'ActualMaps/Actual_{year}.png')
        #input('press key to continue')

    @staticmethod
    def get_hex_grid_mean_C_SOC(year):
        soc_hex_grid = pd.read_csv(r'DataProcessing/soc_hex_grid.csv')
        soc_hex_grid = soc_hex_grid[soc_hex_grid['Year'] == year]

        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        hex_grid['Year'] = year
        
        hex_grid_soc_hex = pd.merge(hex_grid, soc_hex_grid, on=['Hex_ID', 'Year'], how='left')
        mean_bd = np.mean(hex_grid_soc_hex['Hex_Center_BD_x'])
        hex_grid_soc_hex['BulkDensity'] = hex_grid_soc_hex['BulkDensity'].fillna(hex_grid_soc_hex['Hex_Center_BD_x']).fillna(mean_bd)

        soc_hex_mean_targets = hex_grid_soc_hex.groupby(['Hex_ID', 'Hex_Center_Lat_x', 'Hex_Center_Lon_x', 'Year', 'geometry_x']).agg({'C': 'mean', 'BulkDensity': 'mean'}).reset_index().rename(columns={'C': 'Mean_C', 'BulkDensity': 'Mean_BD'})
       
        missing_mean_bd_rows = soc_hex_mean_targets[soc_hex_mean_targets['Mean_BD'].isna()]
        
        if len(missing_mean_bd_rows) > 0:
            raise ValueError(f'Missing mean bulk density. {missing_mean_bd_rows}')
        
        soc_hex_mean_targets = grid_utils.grid_utils.get_geoframe(soc_hex_mean_targets, 'geometry_x')
                                         
        return soc_hex_mean_targets
    
    @staticmethod
    def plot_actual_heat_map():
        soc_hex_grid = pd.read_csv(r'DataProcessing/soc_hex_grid.csv')

        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        
        hex_grid_soc_hex = pd.merge(hex_grid, soc_hex_grid, on=['Hex_ID'], how='left')

        soc_hex_avg_c = grid_utils.grid_utils.get_avg_c_each_grid(df=hex_grid_soc_hex,
                                                        group_by_cols=['Hex_ID', 'Hex_Center_Lat_x', 'Hex_Center_Lon_x', 'geometry_x'],
                                                        avg_col='C',
                                                        avgc_col_rename='Avg_C')
        soc_hex_avg_c = grid_utils.grid_utils.get_geoframe(soc_hex_avg_c, 'geometry_x')
                                         
        map_utils.plot_actual_map(year='(1987-2022)', hex_grid_avg_c=soc_hex_avg_c)
        plot_utils.plot_utils.plot_SOC(model_name='', year_str='(1987-2022)', predictions=soc_hex_grid, map_output_path='MapsTrends/ActualMaps/HeatMap')

    def plot_actual_maps():
        for year in range(1986, 2023):
            hex_avg_grid_C = map_utils.get_hex_grid_mean_C_SOC(year)
            map_utils.plot_actual_map(year=year, hex_grid_avg_c=hex_avg_grid_C)

    @staticmethod
    def plot_predicted_points(year, predictions, predictions_plot_path):
        os.makedirs(os.path.dirname(predictions_plot_path), exist_ok=True)

        if predictions is None or len(predictions) == 0:
            print(f'\nPredictions is missing for year {year}\n')
            return None

        predictions = predictions[predictions['Year']==year]
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        pred_hex_df = grid_utils.grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid,
                                             soil_data=predictions) 

        pred_grid_avg_c_year = grid_utils.grid_utils.get_avg_c_each_grid(df=pred_hex_df,
                                                                   group_by_cols=['Hex_ID', 'geometry', 'Year'],
                                                                   avg_col='C',
                                                                   avgc_col_rename='Avg_C')
        
        grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=pred_grid_avg_c_year,       
                                            geometry_col='geometry',
                                            title=f'Predicted Carbon (% by Mass) for South Africa in Year {year}',
                                            savePlot=True,
                                            output_plot_path=predictions_plot_path)
        #input('press key to continue')

    def plot_combined_predictions_scatter_density_plots(model_name, predictions_folder):
        combined_df = pd.DataFrame()
        for year in range(1986, 2023):
            predictions_path = f'{predictions_folder}/predictions_{year}.csv'
    
            if os.path.exists(predictions_path):
               pred = pd.read_csv(predictions_path)
               pred = pred[pred['C'] > 0]
               combined_df = pd.concat([combined_df, pred], ignore_index=True)

        combined_df.to_csv(os.path.join(predictions_folder, f'combined_predictions.csv'), columns=['Year', 'Lat', 'Lon', 'C', 'Target_C', 'Target_BD', 'SOC'], index=False, mode='w+')
        plot_utils.plot_utils.scatter_plot_predict_c_targetc(df=combined_df, model_name=model_name, output_path=f'{predictions_folder}/scatter_plot.png')
        plot_utils.plot_utils.density_plot_predict_c_targetc(df=combined_df, model_name=model_name, output_path=f'{predictions_folder}/density_plot.png')

'''
map_utils.plot_combined_predictions_scatter_density_plots('CNN', 'MapsTrends/CNN_Model_15360/Predictions')
map_utils.plot_combined_predictions_scatter_density_plots('CNN', 'MapsTrends/CNN_Model_30720/Predictions')
map_utils.plot_combined_predictions_scatter_density_plots('CNN', 'MapsTrends/CNN_Model_61440/Predictions')

map_utils.plot_combined_predictions_scatter_density_plots('RF', 'MapsTrends/RF_Model_15360/Predictions')
map_utils.plot_combined_predictions_scatter_density_plots('RF', 'MapsTrends/RF_Model_30720/Predictions')
map_utils.plot_combined_predictions_scatter_density_plots('RF', 'MapsTrends/RF_Model_61440/Predictions')

trends_analysis.trends_analysis.save_biome_trends('MapsTrends/CNN_Model_15360/Predictions', 'MapsTrends/CNN_Model_15360/Trends')
trends_analysis.trends_analysis.save_biome_trends('MapsTrends/CNN_Model_30720/Predictions', 'MapsTrends/CNN_Model_30720/Trends')
trends_analysis.trends_analysis.save_biome_trends('MapsTrends/CNN_Model_61440/Predictions', 'MapsTrends/CNN_Model_61440/Trends')

trends_analysis.trends_analysis.save_biome_trends('MapsTrends/RF_Model_15360/Predictions', 'MapsTrends/RF_Model_15360/Trends')
trends_analysis.trends_analysis.save_biome_trends('MapsTrends/RF_Model_30720/Predictions', 'MapsTrends/RF_Model_30720/Trends')
trends_analysis.trends_analysis.save_biome_trends('MapsTrends/RF_Model_61440/Predictions', 'MapsTrends/RF_Model_61440/Trends')

'''
map_utils.plot_predicted_points(year=2023, 
                                predictions=pd.read_csv('MapsTrends\CNN_Model_30720\Predictions\combined_predictions.csv'), 
                                predictions_plot_path='MapsTrends\CNN_Model_30720\Predictions\PredictedCarbonMaps\soc_combined_predictions.png')

plot_utils.plot_utils.plot_SOC(model_name='CNN', 
                                year_str=2023,
                                predictions=pd.read_csv('MapsTrends\CNN_Model_30720\Predictions\combined_predictions.csv'),
                                map_output_path='MapsTrends\CNN_Model_30720\Predictions\PredictedMaps\soc_combined_predictions.png')

map_utils.plot_predicted_points(year=2023, 
                                predictions=pd.read_csv('MapsTrends\RF_Model_30720\Predictions\combined_predictions.csv'), 
                                predictions_plot_path='MapsTrends\RF_Model_30720\Predictions\PredictedCarbonMaps\soc_combined_predictions.png')

plot_utils.plot_utils.plot_SOC(model_name='RF', 
                                year_str=2023,
                                predictions=pd.read_csv('MapsTrends\RF_Model_30720\Predictions\combined_predictions.csv'),
                                map_output_path='MapsTrends\RF_Model_30720\Predictions\PredictedMaps\soc_combined_predictions.png')

