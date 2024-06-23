from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_model_utils import base_model_utils
from Model.data_utils import data_utils

def test(model):
    # C% for (-25.8065,27.84747222) = 2.43
    # C% for (-30.30497222, 29.54316667) = 4.27
    lat_lon_pairs = [(-25.8065,27.84747222), (-30.30497222, 29.54316667)]

    for idx in range(len(lat_lon_pairs)):
        lat_lon = [lat_lon_pairs[idx]]
        print(f'\n C % for Lat Lon ({lat_lon}):\n')
        for year in [2008]:
            for month in range(1, 13):
                climate_patch = data_utils.get_climate_patch(year=year, month=month, lat_lon_pairs=lat_lon, patch_size_meters=patch_size_meters_climate)
                landsat_patch = data_utils.get_landsat_patch(year=year, lat_lon_pairs=lat_lon, patch_size_meters=patch_size_meters_landsat)
                terrain_patch = data_utils.get_terrain_patch(patch_size_meters=patch_size_meters_terrain)

                predictions = model.predict(landsat_patch, climate_patch, terrain_patch)
                print(f'\t{lat_lon}; Year_Month: {year}_{month} = {predictions[0]:.2f}\n')

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
