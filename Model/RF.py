import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split

from Model.base_model_utils import base_model_utils

class RF:
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None

    def train(self, landsat_data, climate_data, terrain_data, targets):
        n_samples = landsat_data.shape[0]
    
        # Flatten the image data so that each sample's image data is a single row vector. The new shape will be (n_samples, image_width * image_height * channel).
        landsat_data = landsat_data.reshape(n_samples, -1)
        climate_data = climate_data.reshape(n_samples, -1)
        terrain_data = terrain_data.reshape(n_samples, -1)
    
        X = np.concatenate((landsat_data, climate_data, terrain_data), axis=1)
    
        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, targets, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Initialize Random Forest model
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)

        # Predict on validation set
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics on validation set
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        r2_val = r2_score(y_val, y_pred_val)
        
        print("RF Validation Metrics:\n")
        print(f"Root Mean Squared Error: {rmse_val}")
        print(f"R^2 Score: {r2_val*100:.2f}")
        print()

    
    def predict(self, landsat_data, climate_data, terrain_data):
        # Normalize data
        landsat_data = np.array(landsat_data) 
        climate_data = np.array(climate_data)
        terrain_data = np.array(terrain_data)

        n_samples = landsat_data.shape[0]
    
        # Flatten the image data so that each sample's image data is a single row vector. The new shape will be (n_samples, image_width * image_height * channel).
        landsat_data = landsat_data.reshape(n_samples, -1)
        climate_data = climate_data.reshape(n_samples, -1)
        terrain_data = terrain_data.reshape(n_samples, -1)

        X_test = np.concatenate((landsat_data, climate_data, terrain_data), axis=1)
    
        return self.model.predict(X_test)

