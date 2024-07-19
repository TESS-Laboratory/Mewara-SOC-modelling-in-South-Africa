import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from Model.training_data_utils import training_data_utils

class test_metrics:
    def __init__(self, model):
        self.model = model
    
    def predict(self, year, start_month, end_month, hex_grid_year, 
                patch_size_meters_landsat, 
                patch_size_meters_climate, 
                patch_size_meters_terrain, 
                landsat_bands,
                climate_bands,
                terrain_bands,
                save, output_path, error_output_path):
        predictions_columns = ['Year', 'Lat', 'Lon', 'C', 'Target_C', 'Target_BD', 'SOC']  
        lat_field = 'Hex_Center_Lat_x'
        lon_field = 'Hex_Center_Lon_x'
        predictions_df = pd.DataFrame(columns=predictions_columns)

        lat_lon_pairs = list(set(zip(hex_grid_year[lat_field], hex_grid_year[lon_field])))
        error_logs = []

        print(f'\nStart predictions for total {len(lat_lon_pairs)} in year {year}\n')
        terrain_patches_dict, landsat_patches_dict, climate_patches_dict = training_data_utils.get_all_test_patches(lat_lon_pairs=lat_lon_pairs,
                                                          folder_name='Test',
                                                          years=[year], 
                                                          start_month=start_month,
                                                          end_month=end_month,
                                                          patch_size_meters_landsat=patch_size_meters_landsat,
                                                          patch_size_meters_climate=patch_size_meters_climate,
                                                          patch_size_meters_terrain=patch_size_meters_terrain,
                                                          lat_field=lat_field,
                                                          lon_field=lon_field,
                                                          use_saved_patches=True,
                                                          save_patches=True)
        
        predictions = []
        
        soc_avg_yearly = hex_grid_year[(hex_grid_year['Year'] == year)]
                
        if soc_avg_yearly.empty == True:
            return

        lat_lon_pairs = list(set(zip(soc_avg_yearly[lat_field], soc_avg_yearly[lon_field])))
        pixels_climate = training_data_utils.get_climate_patch_size_pixels(patch_size_meters=patch_size_meters_climate)
        
        for lat, lon in lat_lon_pairs:   
            c_percent = None
            bd = None
            soc_pixel = soc_avg_yearly[(soc_avg_yearly[lat_field] == lat) & (soc_avg_yearly[lon_field] == lon)]
            if not soc_pixel.empty:
                c_percent = np.mean(soc_pixel['Mean_C'].values)
                bd = np.mean(soc_pixel['Mean_BD'].values)

            terrain_patch = terrain_patches_dict.get((lat, lon))
                
            landsat_patch = landsat_patches_dict.get((year, lat, lon))
        
            yearly_climate_sum = np.zeros((pixels_climate,pixels_climate,3))
            no_months = 0

            for month in range(start_month, end_month + 1):
                climate_patch = climate_patches_dict.get((year, month, lat, lon))
                
                if climate_patch is None:
                    continue

                yearly_climate_sum += climate_patch
                no_months += 1

            if c_percent is not None and landsat_patch is None:
                error_logs.append(f"\nlandsat patch for year {year} lat {lat} and lon {lon} is missing")
                continue
            
            if c_percent is not None and yearly_climate_sum is None:
                error_logs.append(f"\nclimate patch for year {year} lat {lat} and lon {lon} is missing")
                continue
        
            if c_percent is not None and terrain_patch is None:
                error_logs.append(f"\nterrain patch for year {year} lat {lat} and lon {lon} is missing")
                continue
            
            if landsat_patch is None or yearly_climate_sum is None or terrain_patch is None:
                continue

            climate_avg_patch = yearly_climate_sum / no_months
            
            landsat_patch_bands = np.round(landsat_patch[:, :, landsat_bands], 2)
            climate_patch_bands = np.round(climate_avg_patch[:, :, climate_bands], 2)
            terrain_patch_bands = np.round(terrain_patch[:, :, terrain_bands], 2)
            
            prediction = self.model.predict(landsat_patch=landsat_patch_bands, climate_patch=climate_patch_bands, terrain_patch=terrain_patch_bands)
            soc = (prediction/100)*bd*20
            predictions.append([year, lat, lon, prediction, c_percent, bd, soc])

        predictions_df = pd.DataFrame(predictions, columns=predictions_columns)
        test_metrics.log_errors(error_output_path=error_output_path, errors=error_logs)
       
        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            predictions_df.to_csv(f'{output_path}', index=False, mode='w+')
        return predictions_df
    
    def log_errors(error_output_path, errors):
        if not os.path.exists(error_output_path):
            os.makedirs(os.path.dirname(error_output_path), exist_ok=True)
        with open(error_output_path, "w") as err:
            errors_str = "\n".join(errors)
            err.write(errors_str)

