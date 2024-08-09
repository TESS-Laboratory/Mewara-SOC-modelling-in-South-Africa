from math import sqrt
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from GoogleStorage import google_storage_service

class RF:
    def __init__(self, cloud_storage, model_path=None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path, cloud_storage=cloud_storage)
            self.model_name = os.path.basename(model_path)
        else:
            self.model = None

    def load_model(self, model_path, cloud_storage):
        if cloud_storage:
            local_file_name = google_storage_service.download_model(model_output_path=model_path)
            self.model = self.load_model_local(local_file_name)
        else:
            self.model = self.load_model_local(model_path)
        return self.model

    def train(self, landsat_train, climate_train, terrain_train, targets_train, landsat_val, climate_val, terrain_val, targets_val, landsat_test, climate_test, terrain_test, targets_test, model_output_path, epochs):
        n_samples_train = landsat_train.shape[0]
        n_samples_val = landsat_val.shape[0]
        n_samples_test = landsat_test.shape[0]
       
        # Flatten the image data so that each sample's image data is a single row vector.
        landsat_train_data = landsat_train.reshape(n_samples_train, -1)
        climate_train_data = climate_train.reshape(n_samples_train, -1)
        terrain_train_data = terrain_train.reshape(n_samples_train, -1)

        landsat_val_data = landsat_val.reshape(n_samples_val, -1)
        climate_val_data = climate_val.reshape(n_samples_val, -1)
        terrain_val_data = terrain_val.reshape(n_samples_val, -1)

        landsat_test_data = landsat_test.reshape(n_samples_test, -1)
        climate_test_data = climate_test.reshape(n_samples_test, -1)
        terrain_test_data = terrain_test.reshape(n_samples_test, -1)
    
        X_train = np.concatenate((landsat_train_data, climate_train_data, terrain_train_data), axis=1)
        X_val = np.concatenate((landsat_val_data, climate_val_data, terrain_val_data), axis=1)
        X_test = np.concatenate((landsat_test_data, climate_test_data, terrain_test_data), axis=1)
    
        # Initialize Random Forest model -- Venter et al. ntree = 500, mtry = sqrt of number of covariates
        # Best parameters found using GridSearchCV:  {'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
        
        best_params = {
            'max_depth': 20,
            'max_features': 'log2',
            'min_samples_leaf': 4,
            'min_samples_split': 2,
            'n_estimators': 200
        }
        
        self.model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        
        # Train the model
        self.model = self.model.fit(X_train, targets_train)

        #self.print_feature_importances(model=self.model)

        # Save the model
        self.save_model(model_output_path)
        print(f'RF model {model_output_path} saved successfully.')
        
        # Predict on training set
        y_pred_train = self.model.predict(X_train)
        
        # Calculate metrics on training set
        mae_train = mean_absolute_error(targets_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(targets_train, y_pred_train))
        r2_train = r2_score(targets_train, y_pred_train)
        
        print("\nRF Training Metrics:")
        print(f"MAE: {mae_train}")
        print(f"RMSE: {rmse_train}")
        print(f"R^2 Score: {r2_train*100:.2f}")

        # Predict on validation set
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics on validation set
        mae_val = mean_absolute_error(targets_val, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(targets_val, y_pred_val))
        r2_val = r2_score(targets_val, y_pred_val)
        
        print("\nRF Validation Metrics:")
        print(f"MAE: {mae_val}")
        print(f"RMSE: {rmse_val}")
        print(f"R^2 Score: {r2_val*100:.2f}")

        # Predict on test set
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics on test set
        mae_test = mean_absolute_error(targets_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(targets_test, y_pred_test))
        r2_test = r2_score(targets_test, y_pred_test)
        
        print("\nRF Test Metrics:")
        print(f"MAE: {mae_test}")
        print(f"RMSE: {rmse_test}")
        print(f"R^2 Score: {r2_test*100:.2f}")

        return r2_test

    def predict(self, landsat_patch, climate_patch, terrain_patch):
        landsat_patch_arr = np.array([landsat_patch]) 
        climate_patch_arr = np.array([climate_patch])
        terrain_patch_arr = np.array([terrain_patch])

        n_samples = landsat_patch_arr.shape[0]
    
        # Flatten the image data so that each sample's image data is a single row vector.
        landsat_patch_arr = landsat_patch_arr.reshape(n_samples, -1)
        climate_patch_arr = climate_patch_arr.reshape(n_samples, -1)
        terrain_patch_arr = terrain_patch_arr.reshape(n_samples, -1)

        X_test = np.concatenate((landsat_patch_arr, climate_patch_arr, terrain_patch_arr), axis=1)
    
        return self.model.predict(X_test)[0]
    
    def print_feature_importances(self, model):
        # Get feature importances
        importances = model.coef_
        
        feature_names = []
        feature_importances = []
        # Print feature importances
        print("Feature Importances:")
        for i, importance in enumerate(importances):
            print(f"Feature {i}: {importance:.4f}")
            feature_names.append(i)
            feature_importances.append(importance)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances in Random Forest Model')
        plt.show()

    def save_model(self, model_output_path):
        if not os.path.exists(model_output_path):
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(self.model, model_output_path)
        print(f"Model saved to {model_output_path}")

    def load_model_local(self, model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def get_model(self):
        return self.model
    
    def get_model_name(self):
        return self.model_name