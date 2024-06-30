import os
import numpy as np
from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_model_utils import base_model_utils
from Model.training_data_utils import training_data_utils
from Maps.maps_utils import map_utils

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

years = [1998, 1999, 2000, 2002, 2007, 2008, 2009, 2010, 2016, 2017, 2018]
#years = [2007, 2008]
start_month = 1
end_month = 12
epochs = 30

use_landsat = True
use_climate = True
use_terrain = True

patch_size_meters_landsat = 30720 # roughly 256*256 pixels
patch_size_meters_climate = 30720 # roughly 7*7 pixels
patch_size_meters_terrain = 30720

def get_training_dataset():
    landsat_data, climate_data, terrain_data, targets = training_data_utils.get_training_data(
        soc_data_path=r'DataProcessing/soc_gdf.csv',
        years=years,
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain
        )
    return np.round(landsat_data, 2), np.round(climate_data, 2), np.round(terrain_data, 2), np.round(targets, 2)

def test(model):
    lat_lon_pairs = [(-25.8065,27.84747222), #Jan 2008 - 2.43
                     (-30.30497222, 29.54316667),#Jan 2008 - 4.27
                     (-30.26368888888889,28.95269722222222), #Jan 2022 - 2.36
                     (-30.264241666666667,28.95255), #Jan 2022 3.0
                     (-30.26485,28.952791666666666), #Jan 2022 2.89
                     (-30.265527777777777,28.952927777777777), #Jan 2022 1.89
                     (-30.261816666666668,28.951472222222222) #Jan 2022 5.07
                     ]

    for idx in range(len(lat_lon_pairs)):
        lat_lon = [lat_lon_pairs[idx]]
        print(f'\n C % for Lat Lon ({lat_lon}):\n')
        for year in [2008, 2018]:
            for month in range(1, 2):
                climate_patch = training_data_utils.get_climate_patches_dict(year=year, month=month, lat_lon_pairs=lat_lon, patch_size_meters=patch_size_meters_climate)
                landsat_patch = training_data_utils.get_landsat_patches_dict(year=year, lat_lon_pairs=lat_lon, patch_size_meters=patch_size_meters_landsat)
                terrain_patch = training_data_utils.get_terrain_patches_dict(lat_lon_pairs=lat_lon, patch_size_meters=patch_size_meters_terrain)

                predictions = model.predict(landsat_patch, climate_patch, terrain_patch)
                print(f'\t{lat_lon}; Year_Month: {year}_{month} = {predictions[0]:.2f}\n')

def train(model, model_output_path):
    print(f'\n Fetching training dataset for years {years}:\n')

    landsat_data, climate_data, terrain_data, targets = get_training_dataset()

    print(f'\n Training {model.__class__.__name__} model:\n')

    history = model.train(landsat_data=landsat_data,
                        climate_data=climate_data,
                        terrain_data=terrain_data,
                        targets=targets,
                        model_output_path=model_output_path,
                        epochs=epochs)
    
    if (history != None):
        base_model_utils.plot_trainin_validation_loss(train_loss=history['loss'], val_loss=history['val_loss'])
        input('Press any key to continue')

def keras_tuner():
    landsat_data, climate_data, terrain_data, targets = get_training_dataset()
    print('\n Tuning CNN Model: \n')
    KerasTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets, epochs=6)

def get_model(model_kind, model_path):
    if model_kind == 'RF':
        return RF(model_path=model_path)
    elif model_kind == 'CNN':
        return CNN(model_path=model_path, use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain)
    
def plot_maps(model_kind, model_path):
    model = get_model(model_kind=model_kind, model_path=model_path)
    for year in years:
        map_utils.create_map(year=year, 
                        start_month=1, 
                        end_month=12,
                        model=model,
                        patch_size_meters_landsat=patch_size_meters_landsat,
                        patch_size_meters_climate=patch_size_meters_climate,
                        patch_size_meters_terrain=patch_size_meters_terrain
                        )

if __name__ == "__main__":
    rf = RF()
    cnn = CNN(use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain)

    variables = f'_L{use_landsat}_C{use_climate}_T{use_terrain}_'
    model_output_cnn = f'Model/CNN_Models/CNN{variables}{epochs}.keras'
    model_output_rf = f'Model/RF_Models/RF{variables}{epochs}'

    '''KerasTuner'''
    #keras_tuner()

    '''CNN'''
    #train(model=cnn, model_output_path=model_output_cnn)
    #cnn_test = CNN(model_path=model_output_cnn, use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain)
    #test(cnn_test)

    '''RF'''
    #train(model=rf, model_output_path=model_output_rf)
    #rf_test = RF(model_path=model_output_rf)
    #test(rf_test)

    #cnn_model = CNN(model_path=model_output_cnn)
    #cnn_model.interpret_shap(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data)

    '''Maps'''
    plot_maps(model_kind='RF', model_path=model_output_rf)
  
    