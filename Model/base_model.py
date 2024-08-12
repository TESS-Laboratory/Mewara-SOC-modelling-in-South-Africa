from math import inf
import os
import numpy as np
import pandas as pd
from Model.base_data_utils import base_data_utils
from MapsTrends.plot_utils import plot_utils

class base_model:
    def test(model, model_predictions_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
        predictions = []
        for i in range(lat_lon_data.shape[0]):
            prediction = model.predict(landsat_data[i], climate_data[i], terrain_data[i])
            target_c = targets[i]
            lat = lat_lon_data[i][0]
            lon = lat_lon_data[i][1]
            predictions.append([lat, lon, prediction, target_c])

        df = pd.DataFrame(predictions, columns=['Lat', 'Lon', 'C', 'Target_C'])
        os.makedirs(os.path.dirname(model_predictions_output_path), exist_ok=True)
        df.to_csv(f'{os.path.dirname(model_predictions_output_path)}/predictions.csv', index=False)
        plot_utils.scatter_plot_predict_c_targetc(df=df, model_name=model.__class__.__name__, output_path=model_predictions_output_path)

    def train(model, model_output_path, epochs, fold, train_data, val_data, test_data):
        print(f"\nTraining {model.__class__.__name__} Fold {fold}\n")
            # For this fold, train the model
        lat_lon_train, landsat_train, climate_train, terrain_train, targets_train = train_data
        lat_lon_val, landsat_val, climate_val, terrain_val, targets_val = val_data
        lat_lon_test, landsat_test, climate_test, terrain_test, targets_test = test_data

        lat_lon_train_df = pd.DataFrame(lat_lon_train, columns=['Lat', 'Lon'])
        os.makedirs('Model/Splits', exist_ok=True)
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
    
    def train_val_test_spatial_split(model_class, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets, epochs, k_fold):
        result_test_r2 = []
        best_test_r2 = -inf
        for fold in range(k_fold):
            train_data, val_data, test_data = base_data_utils.spatial_split(
                                                    lat_lon_data=lat_lon_data,
                                                    landsat_data=landsat_data,
                                                    climate_data=climate_data,
                                                    terrain_data=terrain_data,
                                                    targets=targets)
            test_r2, model = base_model.train(model_class, model_output_path, epochs, fold, train_data, val_data, test_data) 
            if (test_r2 > best_test_r2):
                model_class.save_model(model, model_output_path)
                best_test_r2 = test_r2
            result_test_r2.append(test_r2)

        print(f"\nAverage Training Test Accuracy: {np.mean(result_test_r2)}")
        
    def train_spatial_leave_cluster_out_split(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets, epochs, k_fold):
        result_test_r2 = []
        for fold in range(k_fold):
            train_data, val_data, test_data = base_data_utils.spatial_leave_cluster_out_split(
                                                    lat_lon_data=lat_lon_data,
                                                    landsat_data=landsat_data,
                                                    climate_data=climate_data,
                                                    terrain_data=terrain_data,
                                                    targets=targets)
            test_r2 = base_model.train(model, model_output_path, epochs, fold, train_data, val_data, test_data) 
            result_test_r2.append(test_r2)

        print(f"\nAverage Performance Test Accuracy: {np.mean(result_test_r2)}")
        
       

            
        
            
        
         
           

