from math import sqrt
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

class RF:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
            self.model_name = os.path.basename(model_path)
        else:
            self.model = None

    def train(self, landsat_data, climate_data, terrain_data, targets, model_output_path, epochs):
        n_samples = landsat_data.shape[0]
        n_covariates = landsat_data.shape[3] + climate_data.shape[3] + terrain_data.shape[3]
    
        # Flatten the image data so that each sample's image data is a single row vector.
        landsat_data = landsat_data.reshape(n_samples, -1)
        climate_data = climate_data.reshape(n_samples, -1)
        terrain_data = terrain_data.reshape(n_samples, -1)

        landsat_data = np.round(landsat_data, 2)
        climate_data = np.round(climate_data, 2)
        terrain_data = np.round(terrain_data, 2)
        targets = np.round(targets, 2)

        mtry = int(sqrt(n_covariates))
    
        X = np.concatenate((landsat_data, climate_data, terrain_data), axis=1)
    
        # Split data into training, validation sets and test sets
        X_train, X_val, y_train, y_val = train_test_split(X, targets, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
       
        # Initialize Random Forest model -- Venter et al. ntree = 500, mtry = sqrt of number of covariates
        self.model = RandomForestRegressor(n_estimators=500, max_features=mtry, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)

        # Save the model
        self.save_model(model_output_path)
        print(f'RF model {model_output_path} saved successfully.')
        
        # Predict on training set
        y_pred_train = self.model.predict(X_train)
        
        # Calculate metrics on training set
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2_train = r2_score(y_train, y_pred_train)
        
        print("\nRF Training Metrics:")
        print(f"RMSE: {rmse_train}")
        print(f"R^2 Score: {r2_train*100:.2f}")

        # Predict on validation set
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics on validation set
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        r2_val = r2_score(y_val, y_pred_val)
        
        print("\nRF Validation Metrics:")
        print(f"RMSE: {rmse_val}")
        print(f"R^2 Score: {r2_val*100:.2f}")

        # Predict on test set
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics on test set
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        print("\nRF Test Metrics:")
        print(f"RMSE: {rmse_test}")
        print(f"R^2 Score: {r2_test*100:.2f}")

    def predict(self, landsat_patch_arr, climate_patch_arr, terrain_patch_arr):
        landsat_patch_arr = np.array([landsat_patch_arr]) 
        climate_patch_arr = np.array([climate_patch_arr])
        terrain_patch_arr = np.array([terrain_patch_arr])
        
        landsat_patch_arr = np.round(landsat_patch_arr, 2)
        climate_patch_arr = np.round(climate_patch_arr, 2)
        terrain_patch_arr = np.round(terrain_patch_arr, 2)

        n_samples = landsat_patch_arr.shape[0]
    
        # Flatten the image data so that each sample's image data is a single row vector.
        landsat_patch_arr = landsat_patch_arr.reshape(n_samples, -1)
        climate_patch_arr = climate_patch_arr.reshape(n_samples, -1)
        terrain_patch_arr = terrain_patch_arr.reshape(n_samples, -1)

        X_test = np.concatenate((landsat_patch_arr, climate_patch_arr, terrain_patch_arr), axis=1)
    
        return self.model.predict(X_test)[0]
    
    def print_feature_importances(self, X_train):
        feature_importances = self.model.feature_importances_
        # Get feature names
        n_landsat_features = X_train.shape[3] // 3  # Assuming landsat, climate, and terrain are equally sized
        feature_names = (['landsat_' + str(i) for i in range(n_landsat_features)] +
                         ['climate_' + str(i) for i in range(n_landsat_features)] +
                         ['terrain_' + str(i) for i in range(n_landsat_features)])
        
        # Print feature importances
        for name, importance in zip(feature_names, feature_importances):
            print(f"{name}: {importance:.4f}")

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances in Random Forest Model')
        plt.show()

    def save_model(self, model_output_path):
        joblib.dump(self.model, model_output_path)
        print(f"Model saved to {model_output_path}")

    def load_model(self, model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def get_model(self):
        return self.model
    
    def get_model_name(self):
        return self.model_name