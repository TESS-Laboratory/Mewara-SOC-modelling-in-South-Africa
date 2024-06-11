import tensorflow as tf
from tensorflow.keras import layers, models, metrics, losses
from sklearn.model_selection import train_test_split

class CNN():
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None
        
    def _build_model(self, input_shape_landsat, input_shape_dem, input_shape_climate):
        # Landsat CNN branch
        input_landsat = tf.keras.Input(shape=input_shape_landsat)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_landsat)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)

        # DEM CNN branch
        input_dem = tf.keras.Input(shape=input_shape_dem)
        y = layers.Conv2D(32, (3, 3), activation='relu')(input_dem)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Conv2D(128, (3, 3), activation='relu')(y)
        y = layers.GlobalAveragePooling2D()(y)

        # Climate CNN branch
        input_climate = tf.keras.Input(shape=input_shape_climate)
        z = layers.Conv2D(32, (3, 3), activation='relu')(input_climate)
        z = layers.MaxPooling2D((2, 2))(z)
        z = layers.Conv2D(64, (3, 3), activation='relu')(z)
        z = layers.MaxPooling2D((2, 2))(z)
        z = layers.Conv2D(128, (3, 3), activation='relu')(z)
        z = layers.GlobalAveragePooling2D()(z)

        # Combine branches
        combined = layers.concatenate([x, y, z])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dense(1)(combined)

        # Model
        model = models.Model(inputs=[input_landsat, input_dem, input_climate], outputs=combined)
        model.compile(optimizer='adam', loss=losses.RootMeanSquaredError(), metrics=[metrics.Accuracy(), 
                                                                            metrics.RootMeanSquaredError()])
        model.summary()

    def train(self, landsat_patches, dem_patches, climate_patches, targets, model_output_path):
        # Split data into training and validation sets
        landsat_train, landsat_val, dem_train, dem_val, climate_train, climate_val, targets_train, targets_val = train_test_split(
            landsat_patches, dem_patches, climate_patches, targets, test_size=0.2, random_state=42
        )

        # Train the model
        history = self.model.fit(
            [landsat_train, dem_train, climate_train], targets_train,
            epochs=50, batch_size=32,
            validation_data=([landsat_val, dem_val, climate_val], targets_val))
        
        # Evaluate the model on the validation set
        val_loss, val_mae = self.model.evaluate([landsat_val, dem_val, climate_val], targets_val)
        print(f'Validation Loss: {val_loss}')
        print(f'Validation MAE: {val_mae}')

        self.model.save(model_output_path)

    def predict(self, landsat_patches, dem_patches, climate_patches):
        predictions = self.model.predict([landsat_patches, dem_patches, climate_patches])
        print(predictions)

    

