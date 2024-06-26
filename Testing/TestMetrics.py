import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import pandas as pd
from Model.data_utils import data_utils

class TestMetrics:
    def __init__(self, model_path):
        self.model = self._load_model(model_path=model_path)
        self.model_name = os.path.basename(model_path)

    def _load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        return self.model
    
    def predict(self, soc_grid_path, year, lat_lon_pairs, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, save=False):
        result_columns = ['Year', 'Month', 'Lat', 'Lon', 'Predicted_C', 'Target_C']  
        results = []
        soc_data = pd.read_csv(soc_grid_path)
        
        for idx in range(len(lat_lon_pairs)):
            lat, lon = lat_lon_pairs[idx]
            landsat_patch = data_utils.get_landsat_patch(year=year, lat=lat, lon=lon, patch_size_meters=patch_size_meters_landsat)
            terrain_patch = data_utils.get_terrain_patch(year=year, lat=lat, lon=lon, patch_size_meters=patch_size_meters_terrain)
            for month in range(1, 13):
                target_c = None
                soc = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month) & (soc_data['Hex_Center_Lat'] == lat) & (soc_data['Hex_Center_Lon'] == lon)]
                if soc is not None & soc.any():
                    target_c = soc['C']
                climate_patch = data_utils.get_climate_patch(year=year, month=month, lat=lat, lon=lon, patch_size_meters=patch_size_meters_climate) 
                prediction = self.model.predict([landsat_patch, climate_patch, terrain_patch])
                results.append([year, month, lat, lon, prediction, target_c])

        if save:
            output_path = f'Maps/{self.model_name}/{year}'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data_utils.save_csv(arr = results, column_names= result_columns, output_path=f'{output_path}/predictions.csv')

        return results