import tensorflow as tf
from tensorflow.keras import layers, models, metrics, losses

class CNN():
    
    def train(input_shape_landsat, input_shape_dem, input_shape_climate):
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
