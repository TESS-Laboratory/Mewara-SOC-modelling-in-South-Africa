import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models, metrics, losses

class CNN():
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None
        
    def _build_model(self, input_shape_landsat, input_shape_climate, input_shape_dem):
        # Landsat CNN branch
        input_landsat = layers.Input(shape=input_shape_landsat)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_landsat)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        #x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        #x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Climate CNN branch
        input_climate = layers.Input(shape=input_shape_climate)
        y = layers.Conv2D(32, (3, 3), activation='relu')(input_climate)
        y = layers.MaxPooling2D((2, 2), padding='same')(y)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        #y = keras.layers.MaxPooling2D((2, 2), padding='same')(y)
        #y = keras.layers.Conv2D(128, (3, 3), activation='relu')(y)
        y = layers.GlobalAveragePooling2D()(y)

        # DEM CNN branch
        input_dem = layers.Input(shape=input_shape_dem)
        z = layers.Conv2D(32, (3, 3), activation='relu')(input_dem)
        z = layers.MaxPooling2D((2, 2), padding='same')(z)
        z = layers.Conv2D(64, (3, 3), activation='relu')(z)
        #z = keras.layers.MaxPooling2D((2, 2), padding='same')(z)
        #z = keras.layers.Conv2D(128, (3, 3), activation='relu')(z)
        z = layers.GlobalAveragePooling2D()(z)

        # Combine branches
        combined = layers.concatenate([x, y, z])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(1)(combined)

        # Model
        model = models.Model(inputs=[input_landsat, input_climate, input_dem], outputs=combined)
        model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=[metrics.R2Score(),
                                                                                 metrics.MeanAbsoluteError(), 
                                                                                 metrics.RootMeanSquaredError()])

        model.summary()
        self.model = model

    def train(self, landsat_data, climate_data, terrain_data, targets, model_output_path, epochs):
        # Normalize data
        landsat_data = np.array(landsat_data) 
        climate_data = np.array(climate_data)
        terrain_data = np.array(terrain_data)
        targets = np.array(targets)

        # Split data into training and test sets
        landsat_train, landsat_test, climate_train, climate_test, dem_train, dem_test, targets_train, targets_test = train_test_split(
            landsat_data, climate_data, terrain_data, targets, test_size=0.2, random_state=42)

        # Split data into train and validation sets
        landsat_train, landsat_val, climate_train, climate_val, dem_train, dem_val, targets_train, targets_val = train_test_split(
            landsat_train, climate_train, dem_train, targets_train, test_size=0.2, random_state=42)

        # build model
        self._build_model(input_shape_landsat=landsat_data[0].shape, input_shape_climate=climate_data[0].shape, input_shape_dem=terrain_data[0].shape)
        
        # Train the model
        history = self.model.fit(
            [landsat_train, climate_train, dem_train], targets_train,
            epochs=epochs, batch_size=32,
            validation_data=([landsat_val, climate_val, dem_val], targets_val))
        
        # Evaluate the model on the test set
        test_loss, test_r2, test_mae, test_rmse  = self.model.evaluate([landsat_test, climate_test, dem_test], targets_test)
        print(f"\nTestLoss: {test_loss}; Accuracy: {test_r2}; MAE: {test_mae}; RMSE: {test_rmse}")

        #self.model.save(model_output_path)

    def predict(self, landsat_data, climate_data, terrain_data):
        # Normalize data
        landsat_data = np.array(landsat_data) 
        climate_data = np.array(climate_data)
        terrain_data = np.array(terrain_data)

        landsat_data = np.array(landsat_data)
        climate_data = np.array(climate_data)
        terrain_data = np.array(terrain_data)
        return self.model.predict([landsat_data, climate_data, terrain_data])
