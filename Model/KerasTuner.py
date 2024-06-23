
from keras import layers, models, metrics, losses, optimizers
from keras_tuner import RandomSearch

class KerasTuner:
    @staticmethod
    def create_cnn_branch(input_shape, hp):
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(filters=hp.Int('conv_1_filter', min_value=8, max_value=512, step=16),
                          kernel_size=hp.Choice('conv_1_kernel', values=[2, 3, 5]),
                          activation='relu',
                          padding='same')(input_layer)
        x = layers.MaxPooling2D(
            pool_size=hp.Choice('maxpool_1', values=[2, 3, 5]),
            padding='same'
        )(x)
        x = layers.Conv2D(filters=hp.Int('conv_2_filter', min_value=8, max_value=512, step=16),
                          kernel_size=hp.Choice('conv_2_kernel', values=[2, 3, 5]),
                          activation='relu',
                          padding='same')(x)
        x = layers.MaxPooling2D(
            pool_size=hp.Choice('maxpool_2', values=[2, 3, 5]),
            padding='same'
        )(x)
        x = layers.Conv2D(filters=hp.Int('conv_3_filter', min_value=8, max_value=512, step=16),
                          kernel_size=hp.Choice('conv_3_kernel', values=[2, 3, 5]),
                          activation='relu',
                          padding='same')(x)
        x = layers.MaxPooling2D(
            pool_size=hp.Choice('maxpool_3', values=[2, 3, 5]),
            padding='same'
        )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(
            units=hp.Int('dense_1_units', min_value=8, max_value=512, step=16),
            activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return input_layer, x

    @staticmethod
    def _build_model(hp):
        input_shape = (256,256,4) 
        input_layer, branch = KerasTuner.create_cnn_branch(input_shape=input_shape, hp=hp)

        combined = layers.Dense(units=hp.Int('dense_final_units', min_value=8, max_value=512, step=16), activation='relu')(branch)
        output = layers.Dense(1)(combined)  # Regression output

        model = models.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss=losses.MeanSquaredError(),
                      metrics=[metrics.R2Score(),
                               metrics.MeanAbsoluteError(), 
                               metrics.RootMeanSquaredError()])

        return model

    @staticmethod
    def search(input_data, targets, epochs):
        tuner_search = RandomSearch(KerasTuner._build_model,
                                    objective='val_mean_absolute_error',
                                    max_trials=10,
                                    directory=r'Model\KerasTuner',
                                    project_name='kt2')

        tuner_search.search(input_data, targets, epochs=epochs, validation_split=0.2)

        model = tuner_search.get_best_models(num_models=1)[0]

        print("\n Best Model:\n")
        print(model.summary())

        
