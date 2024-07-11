import os
import numpy as np
import tensorflow as tf
from keras import layers, models, metrics, losses, optimizers
from Model.base_data_utils import base_data_utils
from keras.callbacks import EarlyStopping
from GoogleStorage import google_storage_service

class CNN():
    def __init__(self,  use_landsat, use_climate, use_terrain, cloud_storage, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path, cloud_storage=cloud_storage)
            self.model_name = os.path.basename(model_path)
        else:
            self.model = None

        self.use_landsat = use_landsat
        self.use_climate = use_climate
        self.use_terrain = use_terrain

    def get_model_name(self):
        return self.model_name
    
    def load_model(self, model_path, cloud_storage):
        if cloud_storage:
            local_file_name = google_storage_service.download_model(model_output_path=model_path)
            self.model = tf.keras.models.load_model(local_file_name)
        else:
            self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def create_landsat_terrain_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(376, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(56, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(136, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(280, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        return input_layer, x

    def create_climate_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(72, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(504, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(88, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(472, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(264, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(184, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(168, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(344, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(440, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(328, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(40, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(504, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(424, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(280, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(88, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(440, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(280, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(88, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(440, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(280, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(88, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(440, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(264, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return input_layer, x

    def create_cnn_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return input_layer, x

    def _build_model(self, input_shape_landsat, input_shape_climate, input_shape_terrain):
        inputs = []
        branches = []

        if self.use_landsat:
            # Landsat CNN branch
            landsat_input, landsat_branch = self.create_landsat_terrain_branch(input_shape=input_shape_landsat)
            inputs.append(landsat_input)
            branches.append(landsat_branch)

        if self.use_climate:
            # Climate CNN branch
            climate_input, climate_branch = self.create_climate_branch(input_shape=input_shape_climate)
            inputs.append(climate_input)
            branches.append(climate_branch)

        if self.use_terrain:
            # Terrain CNN branch
            terrain_input, terrain_branch = self.create_landsat_terrain_branch(input_shape=input_shape_terrain)
            inputs.append(terrain_input)
            branches.append(terrain_branch)

        # Combine branches
        concatenated = layers.concatenate(branches)
        x = layers.Dense(168, activation='relu')(concatenated)
        output = layers.Dense(1)(x)  # Regression output
        
        # Model
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.MeanSquaredError(), metrics=[metrics.R2Score(),
                                                                                                              metrics.MeanAbsoluteError(), 
                                                                                                              metrics.RootMeanSquaredError()])

        model.summary()
        return model
        
    def train(self, landsat_train, climate_train, terrain_train, targets_train, landsat_val, climate_val, terrain_val, targets_val, landsat_test, climate_test, terrain_test, targets_test, model_output_path, epochs):
        batch_size = 8

        # build model
        self.model = self._build_model(input_shape_landsat=landsat_train[0].shape, 
                                       input_shape_climate=climate_train[0].shape, 
                                       input_shape_terrain=terrain_train[0].shape)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        train_inputs, val_inputs, test_inputs = self.get_train_val_test_inputs(landsat_train, landsat_val, landsat_test, climate_train, climate_val, climate_test, terrain_train, terrain_val, terrain_test)
        
        # Train the model
        history = self.model.fit(
            train_inputs, targets_train,
            epochs=epochs, batch_size=batch_size, callbacks = [early_stopping],
            validation_data=(val_inputs, targets_val))
        
        base_data_utils.plot_trainin_validation_loss(train_loss=history.history['loss'], val_loss=history.history['val_loss'])
        
        # Evaluate the model on the validation set
        print(f'\nEvaluating Testing Data:\n')
        test_loss, test_r2, test_mae, test_rmse  = self.model.evaluate(test_inputs, targets_test, batch_size=batch_size)
        print(f"\nTestLoss: {test_loss}; TestAccuracy: {test_r2 * 100:.2f}%; TestMAE: {test_mae}; TestRMSE: {test_rmse}\n")

        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        self.model.save(model_output_path, overwrite=True)
        print(f"CNN model '{model_output_path}' saved succesfully.")

        return test_r2

    def get_train_val_test_inputs(self, landsat_train, landsat_val, landsat_test, climate_train, climate_val, climate_test, terrain_train, terrain_val, terrain_test):
        train_inputs = []
        val_inputs = []
        test_inputs = []

        if self.use_landsat:
            train_inputs.append(landsat_train)
            val_inputs.append(landsat_val)
            test_inputs.append(landsat_test)

        if self.use_climate:
            train_inputs.append(climate_train)
            val_inputs.append(climate_val)
            test_inputs.append(climate_test)

        if self.use_terrain:
            train_inputs.append(terrain_train)
            val_inputs.append(terrain_val)
            test_inputs.append(terrain_test)
        return train_inputs,val_inputs,test_inputs

    def predict(self, landsat_patch, climate_patch, terrain_patch):
         # Normalize data
        landsat_patch = np.array([landsat_patch]) 
        climate_patch = np.array([climate_patch]) 
        terrain_patch = np.array([terrain_patch]) 
        
        inputs = []
        if self.use_landsat:
            inputs.append(landsat_patch)
        
        if self.use_climate:
            inputs.append(climate_patch)

        if self.use_terrain:
            inputs.append(terrain_patch)

        return self.model.predict(inputs)[0][0]
