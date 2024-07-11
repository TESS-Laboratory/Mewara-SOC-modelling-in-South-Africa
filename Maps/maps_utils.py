import os
import pandas as pd
from Maps.test_metrics import test_metrics
from Maps.plot_utils import plot_utils
import DataProcessing.grid_utils

class map_utils:
    @staticmethod
    def create_map(year, model, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, landsat_bands, climate_bands, terrain_bands, skip_predictions=False):
        tm = test_metrics(model=model)
        output_dir = f'Maps/{model.get_model_name()}'
        error_output_path = f'{output_dir}/Errors/error_{year}.txt'
        predictions_output_path = f'{output_dir}/Predictions/predictions_{year}.csv'
        predicted_plot_output_path = f'{output_dir}/Predictions/PredictedMaps/predictions_{year}.png'

        hex_grid_avg_c = map_utils.get_hex_grid_avg_c(year)
        
        if not skip_predictions:
            print(f'\nFetching predictions for year {year}\n')
            predictions = tm.predict(year=year,
                    hex_grid_year=hex_grid_avg_c,
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
            
        plot_utils.plot_predictions(model_name=model.__class__.__name__, 
                                    year_str=year,
                                    predictions=predictions,
                                    map_output_path=predicted_plot_output_path)

    @staticmethod
    def plot_actual_map(year, hex_grid_avg_c):
        DataProcessing.grid_utils.grid_utils.plot_heat_map(data_avg_c_color_geometry=hex_grid_avg_c,       
                                                            title=f'Average Carbon (% by Mass) for South Africa in Year {year}',
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
        
        hex_grid_soc_hex = pd.merge(hex_grid, soc_hex_grid, on=['Hex_ID', 'Year'], how='left')

        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_avg_c_each_grid(df=hex_grid_soc_hex,
                                                        group_by_cols=['Hex_ID', 'Hex_Center_Lat_x', 'Hex_Center_Lon_x', 'Year', 'geometry_x'],
                                                        avg_col='C',
                                                        avgc_col_rename='Avg_C')
        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_geoframe(soc_hex_avg_c, 'geometry_x')
                                         
        return soc_hex_avg_c
    
    @staticmethod
    def plot_actual_heat_map():
        soc_hex_grid = pd.read_csv(r'DataProcessing/soc_hex_grid.csv')

        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        
        hex_grid_soc_hex = pd.merge(hex_grid, soc_hex_grid, on=['Hex_ID'], how='left')

        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_avg_c_each_grid(df=hex_grid_soc_hex,
                                                        group_by_cols=['Hex_ID', 'Hex_Center_Lat_x', 'Hex_Center_Lon_x', 'geometry_x'],
                                                        avg_col='C',
                                                        avgc_col_rename='Avg_C')
        soc_hex_avg_c = DataProcessing.grid_utils.grid_utils.get_geoframe(soc_hex_avg_c, 'geometry_x')
                                         
        map_utils.plot_actual_map(year='(1987-2022)', hex_grid_avg_c=soc_hex_avg_c)
        plot_utils.plot_predictions(model_name='', year_str='(1987-2022)', predictions=soc_hex_grid, map_output_path='Maps/ActualMaps/HeatMap')

    def plot_actual_maps():
        for year in range(1986, 2023):
            hex_avg_grid_C = map_utils.get_hex_grid_avg_c(year)
            map_utils.plot_actual_map(year=year, hex_grid_avg_c=hex_avg_grid_C)

    @staticmethod
    def plot_predicted_points(year, predictions, predictions_plot_path):
        os.makedirs(os.path.dirname(predictions_plot_path), exist_ok=True)

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
                                                            title=f'Predicted Carbon (% by Mass) for South Africa in Year {year}',
                                                            savePlot=True,
                                                            output_plot_path=predictions_plot_path)
        #input('press key to continue')

    def plot_predicted_maps_and_scatter_plot(model_name, predictions_folder):
        combined_df = pd.DataFrame()
        for year in range(1986, 2023):
            predictions_path = f'{predictions_folder}/predictions_{year}.csv'
            output_path = f'{predictions_folder}/PredictedMaps/predictions_{year}.png'
    
            if os.path.exists(predictions_path):
               pred = pd.read_csv(predictions_path)
               combined_df = pd.concat([combined_df, pred], ignore_index=True)
               plot_utils.plot_predictions(model_name=model_name, year_str=year, predictions=pred, map_output_path=output_path)

        combined_df.to_csv(os.path.join(predictions_folder, 'combined_predictions.csv'), columns=['Year', 'Lat', 'Lon', 'C', 'Target_C'] )
        plot_utils.scatter_plot_predict_c_targetc(df=combined_df, model_name=model_name, output_path=f'{predictions_folder}/scatter_plot.png')

#map_utils.plot_actual_maps()
#map_utils.plot_actual_heat_map()
#map_utils.plot_predicted_maps_and_scatter_plot(model_name='CNN', predictions_folder='Maps/CNN_Model.keras/Predictions')
#map_utils.plot_predicted_maps_and_scatter_plot(model_name='RF', predictions_folder='Maps/RF_Model/Predictions')
#map_utils.plot_predicted_points(2008, predictions=pd.read_csv('Maps\CNN_Model.keras\Predictions\predictions_2008.csv'), predictions_plot_path='Maps\CNN_Model.keras\Predictions\HexGridMap\predictions_points_2008.png')

