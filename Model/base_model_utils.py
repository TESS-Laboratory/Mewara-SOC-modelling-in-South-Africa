from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

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
    
    def plot_trainin_validation_loss(epochs, train_loss, val_loss):
        epochs = range(1, epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def data_augmentation(X_train):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )

        # Fit the data generator on your training data
        datagen.fit(X_train)

        return datagen

