import random
from tensorflow.keras import layers, models, metrics, losses, optimizers
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

class KerasTuner:
    @staticmethod
    def create_branch(input_shape, name, hp):
        input_layer = layers.Input(shape=input_shape)

        x = input_layer
        for i in range(hp.Int(f'num_layers_{name}', min_value=3, max_value=30, step=2)):
            x = layers.Conv2D(filters=hp.Int(f'conv_{i}_filter_{name}', min_value=32, max_value=512, step=16),
                              kernel_size=(3,3),
                              activation='relu',
                              padding='same')(x)
            x = layers.MaxPooling2D(pool_size=(2,2),
                                    padding='same')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=hp.Int(f'dense_1_units_{name}', min_value=32, max_value=512, step=16),
                         activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        return input_layer, x

    @staticmethod
    def _build_model(hp):
        input_shape_landsat = (256,256,4)
        #input_shape_climate = (4,4,1)
        #input_shape_terrain = (128,128,2)
        landsat_layer, landsat_branch = KerasTuner.create_branch(input_shape=input_shape_landsat, name='landsat', hp=hp)
        #climate_layer, climate_branch = KerasTuner.create_branch(input_shape=input_shape_climate, name='climate', hp=hp)
        #terrain_layer, terrain_branch = KerasTuner.create_branch(input_shape=input_shape_terrain, name='terrain', hp=hp)

        combined_branch = layers.concatenate([landsat_branch])

        combined = layers.Dense(units=hp.Int('dense_final_units', min_value=64, max_value=512, step=8),
                                activation='relu')(combined_branch)
        output = layers.Dense(1)(combined)  # Regression output

        model = models.Model(inputs=[landsat_layer], outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
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

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        tuner_search.search([input_landsat_data, input_climate_data, input_terrain_data], targets, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

        model = tuner_search.get_best_models(num_models=1)[0]

        print("\n Best Model:\n")
        print(model.summary())