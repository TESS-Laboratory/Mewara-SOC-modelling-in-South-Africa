import keras
from sklearn.model_selection import train_test_split

class CNN():
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None
        
    def _build_model(self, landsat_data, climate_data, terrain_data):
        input_shape_landsat = landsat_data[0].shape
        input_shape_climate = climate_data[0].shape
        input_shape_dem = terrain_data[0].shape

        # Landsat CNN branch
        input_landsat = keras.layers.Input(shape=input_shape_landsat)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_landsat)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        #x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        #x = keras.layers.GlobalAveragePooling2D()(x)

        # Climate CNN branch
        input_climate = keras.layers.Input(shape=input_shape_climate)
        y = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_climate)
        y = keras.layers.MaxPooling2D((2, 2), padding='same')(y)
        y = keras.layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = keras.layers.MaxPooling2D((2, 2), padding='same')(y)
        #y = keras.layers.Conv2D(128, (3, 3), activation='relu')(y)
        #y = keras.layers.GlobalAveragePooling2D()(y)

        # DEM CNN branch
        input_dem = keras.layers.Input(shape=input_shape_dem)
        z = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_dem)
        z = keras.layers.MaxPooling2D((2, 2), padding='same')(z)
        z = keras.layers.Conv2D(64, (3, 3), activation='relu')(z)
        z = keras.layers.MaxPooling2D((2, 2), padding='same')(z)
        #z = keras.layers.Conv2D(128, (3, 3), activation='relu')(z)
        #z = keras.layers.GlobalAveragePooling2D()(z)

        # Combine branches
        combined = keras.layers.concatenate([x, y, z])
        combined = keras.layers.Dense(64, activation='relu')(combined)
        combined = keras.layers.Dense(1)(combined)

        # Model
        model = keras.Model(inputs=[input_landsat, input_climate, input_dem], outputs=combined)
        model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.Accuracy(), 
                                                                            keras.metrics.RootMeanSquaredError()])
       
        model.summary()
        self.model = model

    def train(self, landsat_data, climate_data, terrain_data, targets, model_output_path, epochs):
       

        # Split data into training and validation sets
        landsat_train, landsat_val, climate_train, climate_val, dem_train, dem_val, targets_train, targets_val = train_test_split(
            landsat_data, climate_data, terrain_data, targets, test_size=0.2, random_state=42
        )

        # Train the model
        history = self.model.fit(
            [landsat_train, climate_train, dem_train], targets_train,
            epochs=epochs, batch_size=32,
            validation_data=([landsat_val, climate_val, dem_val], targets_val))
        
        # Evaluate the model on the validation set
        val_loss, val_accuracy, val_rmse = self.model.evaluate([landsat_val, climate_val, dem_val], targets_val)
        print(f'Validation Loss: {val_loss}')
        print(f'Validation Accuracy: {val_accuracy}')
        print(f'Validation RMSE: {val_rmse}')

        self.model.save(model_output_path)

    def predict(self, landsat_data, climate_data, terrain_data):
        predictions = self.model.predict([landsat_data, climate_data, terrain_data])
        print(predictions)