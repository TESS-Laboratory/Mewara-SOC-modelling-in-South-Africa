import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from Model.training_data_utils import training_data_utils

class test_metrics:
    def __init__(self, model):
        self.model = model
    
    def predict(self, year, start_month, end_month, hex_grid_year_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, save, output_path, error_output_path):
        predictions_columns = ['Year', 'Month', 'Lat', 'Lon', 'C', 'Target_C']  
        lat_field = 'Hex_Center_Lat'
        lon_field = 'Hex_Center_Lon'
        predictions = []

        lat_lon_pairs_total = list(set(zip(hex_grid_year_month[lat_field], hex_grid_year_month[lon_field])))

        print(f'\nStart predictions for total {len(lat_lon_pairs_total)} in year {year}\n')
        terrain_patches_dict, landsat_patches_dict, climate_patches_dict = training_data_utils.get_patches(soc_data=hex_grid_year_month,
                                                          folder_name='Test',
                                                          years=[year], 
                                                          start_month=start_month,
                                                          end_month=end_month,
                                                          patch_size_meters_landsat=patch_size_meters_landsat,
                                                          patch_size_meters_climate=patch_size_meters_climate,
                                                          patch_size_meters_terrain=patch_size_meters_terrain,
                                                          lat_field=lat_field,
                                                          lon_field=lon_field)
       
        for month in range(start_month, end_month + 1):
            soc_avg_monthly = hex_grid_year_month[(hex_grid_year_month['Year'] == year) & (hex_grid_year_month['Month'] == month)]
                
            if soc_avg_monthly.empty == True:
                continue

            lat_lon_pairs = list(set(zip(soc_avg_monthly[lat_field], soc_avg_monthly[lon_field])))

            for lat, lon in lat_lon_pairs:
                c_percent = None
                soc = soc_avg_monthly[(soc_avg_monthly[lat_field] == lat) & 
                                       (soc_avg_monthly[lon_field] == lon)]
                if not soc.empty:
                    c_percent = soc['Avg_C'].values[0]
                    c_percent = soc['Avg_C'].values[0]

                terrain_patch = terrain_patches_dict.get((lat, lon))
                    
                landsat_patch = landsat_patches_dict.get((year, lat, lon))
            
                climate_patch = climate_patches_dict.get((year, month, lat, lon))
               
                if c_percent is not None and landsat_patch is None:
                    test_metrics.log_error(error_output_path=error_output_path,
                                           error_text=f"\nlandsat patch for year {year} month {month} lat {lat} and lon {lon} is missing")
                
                if c_percent is not None and climate_patch is None:
                    test_metrics.log_error(error_output_path=error_output_path,
                                           error_text=f"\nclimate patch for year {year} month {month} lat {lat} and lon {lon} is missing")
                
                if c_percent is not None and terrain_patch is None:
                    test_metrics.log_error(error_output_path=error_output_path,
                                           error_text=f"\nterrain patch for year {year} month {month} lat {lat} and lon {lon} is missing")
                
                if landsat_patch is None or climate_patch is None or terrain_patch is None:
                    prediction = None
                    continue
                else:
                    prediction = self.model.predict(landsat_patch=landsat_patch, climate_patch=climate_patch, terrain_patch=terrain_patch)
                
                predictions.append([year, month, lat, lon, prediction, c_percent])
                predictions.append([year, month, lat, lon, prediction, c_percent])

        predictions_df = pd.DataFrame(predictions, columns=predictions_columns)

        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            training_data_utils.save_csv(arr = predictions, column_names= predictions_columns, output_path=f'{output_path}')
        return predictions_df
    
    def log_error(error_output_path, error_text):
        if not os.path.exists(error_output_path):
            os.makedirs(os.path.dirname(error_output_path), exist_ok=True)
        with open(error_output_path, "a") as err:
            err.write(error_text)