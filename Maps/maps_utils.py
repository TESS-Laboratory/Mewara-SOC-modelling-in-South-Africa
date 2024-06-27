import pandas as pd
from Maps.test_metrics import test_metrics
from DataProcessing.grid_utils import grid_utils

class map_utils:
    @staticmethod
    def create_map(year, start_month, end_month, model, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain):
        tm = test_metrics(model=model)
        output_dir = f'Maps/{model.get_model_name()}'
        predictions_output_path = f'{output_dir}/predictions_{year}.csv'
        predicted_plot_output_path = f'{output_dir}/Predicted/predicted_{year}.png'

        hex_grid = pd.read_csv(r'DataProcessing/soc_hex_grid.csv')
        hex_grid = hex_grid.dropna(subset=['Hex_Center_Lat', 'Hex_Center_Lon'])
        lat_lon_pairs = list(set(zip(hex_grid['Hex_Center_Lat'], hex_grid['Hex_Center_Lon'])))

        print(f'\nFetching predictions for year {year}\n')
        predictions = tm.predict(year=year,
                   start_month=start_month,
                   end_month=end_month,
                   lat_lon_pairs=lat_lon_pairs, 
                   soc_path=r'DataProcessing/soc_hex_grid.csv',
                   patch_size_meters_landsat=patch_size_meters_landsat,
                   patch_size_meters_climate=patch_size_meters_climate, patch_size_meters_terrain=patch_size_meters_terrain,
                   save=True,
                   output_path=predictions_output_path)
        map_utils.plot_actual_map(year)
        map_utils.plot_predicted_map(year=year, 
                                     predictions=predictions,
                                     predictions_plot_path=predicted_plot_output_path)

    @staticmethod
    def plot_actual_map(year):
        soil_data = pd.read_csv(r'DataProcessing/soc_gdf.csv')
        soil_data_year = soil_data[soil_data['Year'] == year]
        grid_utils.plot_soil_data_heat_map(soil_data=soil_data_year, 
                                           title=f'Average C (% by Mass) for South Africa in year {year}',
                                           use_square_grid=False,
                                           savePlot=True,
                                           output_plot_path=f'Maps/ActualMaps/Actual_{year}.png')
        #input('press key to continue')
        
    @staticmethod
    def plot_predicted_map(year, predictions, predictions_plot_path):
        if predictions is None or len(predictions) == 0:
            print(f'\nPredictions is missing for year {year}\n')
            return None
        
        predictions_year = predictions[predictions['Year'] == year]
        grid_utils.plot_soil_data_heat_map(soil_data=predictions_year, 
                                           title=f'Predicted Average C (% by Mass) for South Africa in year {year}',
                                           use_square_grid=False,
                                           savePlot=True,
                                           output_plot_path=predictions_plot_path)
        #input('press key to continue')