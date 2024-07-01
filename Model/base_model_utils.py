import os
import pickle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from Model.training_data_utils import training_data_utils

class base_model_utils:
    def get_train_val_test_data(landsat_data, climate_data, terrain_data, targets):
        # Split data into training and test sets
        landsat_train, landsat_test, climate_train, climate_test, dem_train, dem_test, targets_train, targets_test = train_test_split(
            landsat_data, climate_data, terrain_data, targets, test_size=0.2, random_state=42)

        # Split data into train and validation sets
        landsat_train, landsat_val, climate_train, climate_val, dem_train, dem_val, targets_train, targets_val = train_test_split(
            landsat_train, climate_train, dem_train, targets_train, test_size=0.2, random_state=42)
        
        return landsat_train, landsat_val, landsat_test, climate_train, climate_val, climate_test, dem_train, dem_val, dem_test, \
        targets_train, targets_val, targets_test
    
    def get_training_data_with_cache(cache_path, soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain):
        # Check if cache exists
        if os.path.exists(cache_path):
            print(f"Loading data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data['landsat_data'], data['climate_data'], data['terrain_data'], data['targets']
        
        # Fetch the data
        print("Fetching data from source")
        landsat_data, climate_data, terrain_data, targets = training_data_utils.get_training_data(
            soc_data_path=soc_data_path,
            years=years,
            start_month=start_month,
            end_month=end_month,
            patch_size_meters_landsat=patch_size_meters_landsat,
            patch_size_meters_climate=patch_size_meters_climate,
            patch_size_meters_terrain=patch_size_meters_terrain
        )

        # Save the data to cache
        print(f"Saving data to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'landsat_data': landsat_data,
                'climate_data': climate_data,
                'terrain_data': terrain_data,
                'targets': targets
            }, f)

        return landsat_data, climate_data, terrain_data, targets

    def plot_trainin_validation_loss(train_loss, val_loss):
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()