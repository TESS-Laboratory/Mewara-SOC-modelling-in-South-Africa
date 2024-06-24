import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from matplotlib import pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models, metrics, losses, optimizers
from Model.base_model_utils import base_model_utils
from keras.callbacks import EarlyStopping

class CNN():
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        return self.model
    
    def create_landsat_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(filters=184, kernel_size=(5,5), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D(pool_size=(3,3), padding='same')(x)
        
        x = layers.Conv2D(filters=456, kernel_size=(5,5), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(5,5), padding='same')(x)
        
        x = layers.Conv2D(filters=56, kernel_size=(2,2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(units=328, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(units=88, activation='relu')(x)
        
        return input_layer, x
    
    def create_climate_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(filters=328, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D(pool_size=(3,3), padding='same')(x)
        
        x = layers.Conv2D(filters=344, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        
        x = layers.Conv2D(filters=312, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(units=104, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(units=168, activation='relu')(x)
        
        return input_layer, x
    
    def create_cnn_branch(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return input_layer, x

    def _build_model(self, input_shape_landsat, input_shape_climate, input_shape_terrain):

        # Landsat CNN branch
        landsat_input, landsat_branch = self.create_landsat_branch(input_shape=input_shape_landsat)

        # Climate CNN branch
        climate_input, climate_branch = self.create_climate_branch(input_shape=input_shape_climate)

        # Terrain CNN branch
        terrain_input, terrain_branch = self.create_cnn_branch(input_shape=input_shape_terrain)

        # Combine branches
        combined = layers.concatenate([landsat_branch, climate_branch, terrain_branch])

        # Fully connected layers and regression head
        output = layers.Dense(1)(combined)  # Regression output
        
        # Model
        model = models.Model(inputs=[landsat_input, climate_input, terrain_input], outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.MeanSquaredError(), metrics=[metrics.R2Score(),
                                                                                                               metrics.MeanAbsoluteError(), 
                                                                                                               metrics.RootMeanSquaredError()])

        model.summary()
        return model
        
    def train(self, landsat_data, climate_data, terrain_data, targets, model_output_path, epochs):
        # Normalize data
        landsat_data = np.array(landsat_data) 
        climate_data = np.array(climate_data) 
        terrain_data = np.array(terrain_data) 
        targets = np.array(targets)

        landsat_data = np.round(landsat_data, 2)
        climate_data = np.round(climate_data, 2)
        terrain_data = np.round(terrain_data, 2)
        targets = np.round(targets, 2)

        # Split data into training and test sets
        landsat_train, landsat_val, landsat_test, climate_train, climate_val, climate_test, \
              terrain_train, terrain_val, terrain_test, targets_train, targets_val, targets_test \
            = base_model_utils.get_train_val_test_data(landsat_data=landsat_data,
                                                  climate_data=climate_data,
                                                  terrain_data=terrain_data,
                                                  targets=targets)
       
        # build model
        self.model = self._build_model(input_shape_landsat=landsat_data[0].shape, input_shape_climate=climate_data[0].shape)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            [landsat_train, climate_train], targets_train,
            epochs=epochs, batch_size=32, callbacks = [early_stopping],
            validation_data=([landsat_val, climate_val], targets_val))
        
        # Evaluate the model on the validation set
        print(f'\nEvaluating Testing Data:\n')
        test_loss, test_r2, test_mae, test_rmse  = self.model.evaluate([landsat_test, climate_test], targets_test)
        print(f"\nTestLoss: {test_loss}; TestAccuracy: {test_r2 * 100:.2f}%; TestMAE: {test_mae}; TestRMSE: {test_rmse}\n")

        self.model.save(model_output_path, overwrite=True)
        print(f'CNN model \'{model_output_path}\' saved succesfully.')

        return history.history

    def predict(self, landsat_data, climate_data, terrain_data):
        # Normalize data
        landsat_data = np.array(landsat_data) 
        climate_data = np.array(climate_data) 
        terrain_data = np.array(terrain_data) 

        landsat_data = np.round(landsat_data, 2)
        climate_data = np.round(climate_data, 2)
        terrain_data = np.round(terrain_data, 2)
        targets = np.round(targets, 2)

        return self.model.predict([landsat_data, climate_data])[0]

    def interpret_shap(self, landsat_data, climate_data, terrain_data, num_samples=100):
        # Select 100 samples from each input data
        landsat_samples = landsat_data[:num_samples]
        #climate_samples = climate_data[:num_samples]
        #terrain_samples = terrain_data[:num_samples] / 255
        
        # Concatenate landsat, climate, and terrain data along the last axis
        #X_test = np.concatenate([landsat_samples, climate_samples], axis=-1)
        
        # Initialize the SHAP explainer with your Keras model predict function
        explainer = shap.DeepExplainer(self.model, [landsat_samples])
        
        # Compute SHAP values
        shap_values = explainer.shap_values([landsat_samples])

        # Check the shape of SHAP values
        print(f'SHAP values shape: {np.array(shap_values).shape}')

        # Reshape SHAP values to match input data shape
        shap_values = np.array(shap_values).reshape((num_samples, 256, 256, 4))
        
        # Aggregate SHAP values across all samples
        shap_values_agg = np.mean(np.abs(shap_values), axis=(0,1,2))  # Use mean absolute value for aggregation

        channel_ranking = np.argsort(shap_values_agg)[::-1]

        # Print channel importance ranking
        print("Channel Importance Ranking:")
        for rank, channel_idx in enumerate(channel_ranking):
            print(f"Rank {rank + 1}: Channel {channel_idx}")

        # Plot the aggregated SHAP values for each channel
        fig, ax = plt.subplots()
        ax.barh(range(4), shap_values_agg[channel_ranking], align='center', color='skyblue')
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'Channel {idx}' for idx in channel_ranking])
        ax.invert_yaxis()  # Highest importance at the top
        ax.set_xlabel('SHAP Importance')
        ax.set_title('Channel Importance Ranking')
        plt.show()