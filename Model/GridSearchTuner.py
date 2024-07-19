from math import sqrt
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

class GridSearchTuner:
    def search(input_landsat_data, input_climate_data, input_terrain_data, targets):
        n_samples = input_landsat_data.shape[0]
        n_covariates = input_landsat_data.shape[3] + input_climate_data.shape[3] + input_terrain_data.shape[3]
    
        # Flatten the image data so that each sample's image data is a single row vector.
        landsat_data = input_landsat_data.reshape(n_samples, -1)
        climate_data = input_climate_data.reshape(n_samples, -1)
        terrain_data = input_terrain_data.reshape(n_samples, -1)

        X = np.concatenate((landsat_data, climate_data, terrain_data), axis=1)
    
        param_grid = {
        'n_estimators': [100, 200, 500],
        'max_features': [2, 3, 5, int(sqrt(n_covariates)), 'auto', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
    
        # Initialize the RandomForestRegressor
        rf = RandomForestRegressor(random_state=42)
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        
        # Fit GridSearchCV
        grid_search.fit(X, targets)
        
        # Best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print("Best parameters found: ", best_params)
        
        return best_model
    
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