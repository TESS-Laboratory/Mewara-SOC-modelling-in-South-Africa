import numpy as np
import pandas as pd
from Model.base_data_utils import base_data_utils

class base_model:
    
    def get_train_val_test_sets(model, model_output_path, epochs, fold, train_data, val_data, test_data):
        print(f"\nTraining {model.__class__.__name__} Fold {fold}\n")
            # For this fold, train the model
        lat_lon_train, landsat_train, climate_train, terrain_train, targets_train = train_data
        lat_lon_val, landsat_val, climate_val, terrain_val, targets_val = val_data
        lat_lon_test, landsat_test, climate_test, terrain_test, targets_test = test_data

        lat_lon_train_df = pd.DataFrame(lat_lon_train, columns=['Lat', 'Lon'])
        lat_lon_train_df.to_csv(f'Model/Splits/train_{fold}.csv', index=False)

        lat_lon_val_df = pd.DataFrame(lat_lon_val, columns=['Lat', 'Lon'])
        lat_lon_val_df.to_csv(f'Model/Splits/val_{fold}.csv', index=False)

        lat_lon_test_df = pd.DataFrame(lat_lon_test, columns=['Lat', 'Lon'])
        lat_lon_test_df.to_csv(f'Model/Splits/test_{fold}.csv', index=False)

        print(f"\n Training with {len(lat_lon_train)} train, {len(lat_lon_val)} val and {len(lat_lon_test)} test data")

        return model.train(landsat_train = landsat_train, 
                        landsat_val = landsat_val, 
                        landsat_test = landsat_test, 
                        climate_train = climate_train, 
                        climate_val = climate_val, 
                        climate_test = climate_test, 
                        terrain_train = terrain_train, 
                        terrain_val = terrain_val, 
                        terrain_test = terrain_test, 
                        targets_train = targets_train, 
                        targets_val = targets_val, 
                        targets_test = targets_test, 
                        model_output_path = model_output_path, 
                        epochs = epochs)
    
    def train_val_test_spatial_split(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets, epochs, k_fold):
        result_test_r2 = []
        for fold in range(k_fold):
            train_data, val_data, test_data = base_data_utils.spatial_leave_cluster_out_split(
                                                    lat_lon_data=lat_lon_data,
                                                    landsat_data=landsat_data,
                                                    climate_data=climate_data,
                                                    terrain_data=terrain_data,
                                                    targets=targets)
            test_r2 = model.get_train_val_test_sets(model_output_path, epochs, fold, train_data, val_data, test_data) 
            result_test_r2.append(test_r2)

        print(f"\nAverage Test Accuracy: {np.mean(result_test_r2)}")
        
    def train_spatial_kfold(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets, epochs, k_fold):
        result_test_r2 = []
        for fold in range(k_fold):
            train_data, val_data, test_data = base_data_utils.spatial_leave_cluster_out_split(
                                                    lat_lon_data=lat_lon_data,
                                                    landsat_data=landsat_data,
                                                    climate_data=climate_data,
                                                    terrain_data=terrain_data,
                                                    targets=targets)
            test_r2 = model.get_train_val_test_sets(model_output_path, epochs, fold, train_data, val_data, test_data) 
            result_test_r2.append(test_r2)

        print(f"\nAverage Test Accuracy: {np.mean(result_test_r2)}")
        
       

            
        
            
        
         
           

