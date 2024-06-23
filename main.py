from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_model_utils import base_model_utils
from Model.data_utils import data_utils


def train(model, model_output_path):
    print(f'\n Training {model.__class__.__name__} model:\n')

    history = model.train(landsat_data=landsat_data,
                        climate_data=climate_data,
                        terrain_data=terrain_data,
                        targets=targets,
                        model_output_path = model_output_path,
                        epochs=epochs)
    
    if (history != None):
        base_model_utils.plot_trainin_validation_loss(epochs=epochs, train_loss=history['loss'], val_loss=history['val_loss'])
        input('Press any key to continue')

def predict(model):
   # C% for (-25.8065,27.84747222) = 2.43
    # C% for (-30.30497222, 29.54316667) = 4.27
    lat_lon_pairs = [(-25.8065,27.84747222), (-30.30497222, 29.54316667)]
    
def grid_search_CNN():
    cnn = CNN()
    landsat_data, climate_data, terrain_data, targets = data_utils.get_training_data(
        soc_data_path=r'DataProcessing\soc_gdf.csv',
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain
        )
    best_model = cnn.grid_search(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    best_model.save('Model\CNN_Models\bestModel.keras')

def keras_tuner():
    KerasTuner.search(input_data=landsat_data, targets=targets, epochs=6)

years = [2000, 2007, 2008, 2009, 2017, 2018]
start_month = 1
end_month = 12
epochs = 10

patch_size_meters_landsat = 30720 # roughly 256*256 pixels
patch_size_meters_climate = 30720 # roughly 7*7 pixels
patch_size_meters_terrain = 30720

landsat_data, climate_data, terrain_data, targets = data_utils.get_training_data(
        soc_data_path=r'DataProcessing\soc_gdf.csv',
        years=years,
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain
        )

if __name__ == "__main__":
    rf = RF()
    cnn = CNN()

    variables = '_Landsat_Climate_'
    model_output_cnn = f'Model\{cnn.__class__.__name__}_Models\{cnn.__class__.__name__}{variables}{epochs}.keras'
    model_output_rf = f'Model\{rf.__class__.__name__}_Models\{rf.__class__.__name__}{variables}{epochs}'

    '''KerasTuner'''
    keras_tuner()

    '''CNN'''
    #train(model=cnn, model_output_path=model_output_cnn)
    #cnn_test = CNN(model_path=model_output_cnn)
    #test(cnn_test)
    
    '''RF'''
    #train(model=rf, model_output_path=model_output_rf)
    #rf_test = RF(model_path=model_output_rf)
    #test(rf_test)

    #cnn_model = CNN(model_path=model_output_cnn)
    #cnn_model.interpret_shap(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data)
