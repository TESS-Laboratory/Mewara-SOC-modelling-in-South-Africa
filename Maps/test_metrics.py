import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from Model.training_data_utils import training_data_utils

class test_metrics:
    def __init__(self, model):
        self.model = model
    
    def predict(self, year, start_month, end_month, lat_lon_pairs, soc_path, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, save=False, output_path=''):
        predictions_columns = ['Year', 'Month', 'Lat', 'Lon', 'C', 'Target_C']  
        predictions = []
        soc_data = pd.read_csv(soc_path)

        landsat_patches = training_data_utils.get_landsat_patches_dict(year=year, lat_lon_pairs=lat_lon_pairs, patch_size_meters=patch_size_meters_landsat)
        if landsat_patches is None:
            print(f'\nSkipping year {year} as missing landsat raster\n')
            return
        
        terrain_patches = training_data_utils.get_terrain_patches_dict(lat_lon_pairs=lat_lon_pairs, patch_size_meters=patch_size_meters_terrain)
        
        for month in range(start_month, end_month + 1):
            climate_patches = training_data_utils.get_climate_patches_dict(year=year, month=month, lat_lon_pairs=lat_lon_pairs, patch_size_meters=patch_size_meters_climate) 

            for idx in range(len(lat_lon_pairs)):
                lat, lon = lat_lon_pairs[idx]
                landsat_patch = landsat_patches[idx]
                climate_patch = climate_patches[idx]
                terrain_patch = terrain_patches[idx]

                target_c = None
                soc = soc_data[(soc_data['Year'] == year) & 
                               (soc_data['Month'] == month) & 
                               (soc_data['Hex_Center_Lat'] == lat) & 
                               (soc_data['Hex_Center_Lon'] == lon)]
                if not soc.empty:
                    target_c = soc['C']

                if landsat_patch is None or climate_patch is None or terrain_patch is None:
                    prediction = None
                else:
                    prediction = self.model.predict(landsat_patch=landsat_patch, climate_patch=climate_patch, terrain_patch=terrain_patch)
                
                if target_c is None and prediction is None:
                    continue

                predictions.append([year, month, lat, lon, prediction, target_c])

        predictions_df = pd.DataFrame(predictions, columns=predictions_columns)

        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            training_data_utils.save_csv(arr = predictions, column_names= predictions_columns, output_path=f'{output_path}')
        return predictions_df