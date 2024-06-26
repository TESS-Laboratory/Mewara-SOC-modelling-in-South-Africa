
import random
from tensorflow.keras import layers, models, metrics, losses, optimizers
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

class KerasTuner:
    @staticmethod
    def create_landsat_branch(input_shape, hp):
        input_layer = layers.Input(shape=input_shape)
        name = 'landsat'
        x = input_layer
        for i in range(hp.Int(f'num_layers_{name}', min_value=2, max_value=20, step=1)):
            x = layers.Conv2D(filters=hp.Int(f'conv_{i}_filter_{name}', min_value=8, max_value=512, step=16),
                              kernel_size=hp.Choice(f'conv_{i}_kernel_{name}', values=[2, 3, 5]),
                              activation='relu',
                              padding='same')(x)
            x = layers.MaxPooling2D(pool_size=hp.Choice(f'maxpool_{i}_{name}', values=[2, 3, 5]), padding='same')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(units=hp.Int(f'dense_1_units_{name}', min_value=8, max_value=512, step=16),
                         activation=hp.Choice(f'activation_{name}', values=['relu', 'linear', 'tanh']))(x)
        x = layers.Dropout(rate=hp.Choice(f'dropout_{name}', values=[0.2, 0.3, 0.5]))(x)
        return input_layer, x
    
    @staticmethod
    def create_climate_branch(input_shape, hp):
        input_layer = layers.Input(shape=input_shape)
        name = 'climate'
        x = input_layer
        for i in range(hp.Int(f'num_layers_{name}', min_value=2, max_value=20, step=1)):
            x = layers.Conv2D(filters=hp.Int(f'conv_{i}_filter_{name}', min_value=8, max_value=512, step=16),
                              kernel_size=hp.Choice(f'conv_{i}_kernel_{name}', values=[2, 3, 5]),
                              activation='relu',
                              padding='same')(x)
            x = layers.MaxPooling2D(pool_size=hp.Choice(f'maxpool_{i}_{name}', values=[2, 3, 5]), padding='same')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(units=hp.Int(f'dense_1_units_{name}', min_value=8, max_value=512, step=16),
                         activation=hp.Choice(f'activation_{name}', values=['relu', 'linear', 'tanh']))(x)
        x = layers.Dropout(rate=hp.Choice(f'dropout_{name}', values=[0.2, 0.3, 0.5]))(x)
        return input_layer, x
    
    @staticmethod
    def create_terrain_branch(input_shape, hp):
        input_layer = layers.Input(shape=input_shape)
        name = 'terrain'
        x = input_layer
        for i in range(hp.Int(f'num_layers_{name}', min_value=2, max_value=20, step=1)):
            x = layers.Conv2D(filters=hp.Int(f'conv_{i}_filter_{name}', min_value=8, max_value=512, step=16),
                              kernel_size=hp.Choice(f'conv_{i}_kernel_{name}', values=[2, 3, 5]),
                              activation='relu',
                              padding='same')(x)
            x = layers.MaxPooling2D(pool_size=hp.Choice(f'maxpool_{i}_{name}', values=[2, 3, 5]), padding='same')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(units=hp.Int(f'dense_1_units_{name}', min_value=8, max_value=512, step=16),
                         activation=hp.Choice(f'activation_{name}', values=['relu', 'linear', 'tanh']))(x)
        x = layers.Dropout(rate=hp.Choice(f'dropout_{name}', values=[0.2, 0.3, 0.5]))(x)
        return input_layer, x

    @staticmethod
    def _build_model(hp):
        input_shape_landsat = (256,256,4) 
        input_shape_climate = (7,7,3)
        input_shape_terrain = (256,256,2)
        landsat_layer, landsat_branch = KerasTuner.create_landsat_branch(input_shape=input_shape_landsat, hp=hp)
        climate_layer, climate_branch = KerasTuner.create_climate_branch(input_shape=input_shape_climate, hp=hp)
        terrain_layer, terrain_branch = KerasTuner.create_terrain_branch(input_shape=input_shape_terrain, hp=hp)

        combined_branch = layers.concatenate([landsat_branch, climate_branch, terrain_branch])

        combined = layers.Dense(units=hp.Int('dense_final_units', min_value=8, max_value=512, step=16), 
                                activation=hp.Choice('activation', values=['relu', 'linear', 'tanh']))(combined_branch)
        output = layers.Dense(1)(combined)  # Regression output

        model = models.Model(inputs=[landsat_layer, climate_layer, terrain_layer], outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss=losses.MeanSquaredError(),
                      metrics=[metrics.R2Score(),
                               metrics.MeanAbsoluteError(), 
                               metrics.RootMeanSquaredError()])

        return model

    @staticmethod
    def search(input_landsat_data, input_climate_data, input_terrain_data, targets, epochs):
        tuner_search = RandomSearch(KerasTuner._build_model,
                                    objective='val_mean_absolute_error',
                                    max_trials=20,
                                    directory=r'Model/KerasTuner',
                                    project_name=f'kt_{random.randint(0, 100)}')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        tuner_search.search([input_landsat_data, input_climate_data, input_terrain_data], targets, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

        model = tuner_search.get_best_models(num_models=1)[0]

        print("\n Best Model:\n")
        print(model.summary())

        
