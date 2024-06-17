from sklearn.model_selection import train_test_split

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
