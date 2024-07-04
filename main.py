import os
import numpy as np
from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_data_utils import base_data_utils
from Maps.maps_utils import map_utils
from Model.data_analysis import data_analysis

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

years = [1998, 1999, 2000, 2001, 2002, 2004, 2007, 2008, 2009, 2010, 2012, 2016, 2017, 2018]
#years=[1998, 1999]
start_month = 1
end_month = 12
epochs = 30

use_landsat = True
use_climate = True
use_terrain = True

patch_size_meters_landsat = 30720 # roughly 256*256 pixels
patch_size_meters_climate = 30720 # roughly 7*7 pixels
patch_size_meters_terrain = 30720

training_soc_path = r'DataProcessing/soc_gdf.csv'

def get_training_dataset():
    landsat_data, climate_data, terrain_data, targets = base_data_utils.get_test_train_data(
        soc_data_path=training_soc_path,
        years=years,
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain)   
    return np.round(landsat_data, 3), np.round(climate_data, 3), np.round(terrain_data, 3), np.round(targets, 3)

def train(model, model_output_path, landsat_data, climate_data, terrain_data, targets):
    print(f'\n Training {model.__class__.__name__} model:\n')
    history = model.train(landsat_data=landsat_data,
                        climate_data=climate_data,
                        terrain_data=terrain_data,
                        targets=targets,
                        model_output_path=model_output_path,
                        epochs=epochs)
    
    if (history != None):
        base_data_utils.plot_trainin_validation_loss(train_loss=history['loss'], val_loss=history['val_loss'])
        #input('Press any key to continue')

def keras_tuner():
    landsat_data, climate_data, terrain_data, targets = get_training_dataset()
    print('\n Tuning CNN Model: \n')
    KerasTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets, epochs=6)

def get_model(model_kind, model_path, cloud_storage):
    if model_kind == 'RF':
        return RF(model_path=model_path)
    elif model_kind == 'CNN':
        return CNN(model_path=model_path, use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage)
    
def plot_maps(model_kind, model_path, cloud_storage):
    model = get_model(model_kind=model_kind, model_path=model_path, cloud_storage=cloud_storage)
    for year in [2008, 2018]:
        map_utils.create_map(year=year, 
                        start_month=1, 
                        end_month=12,
                        model=model,
                        patch_size_meters_landsat=patch_size_meters_landsat,
                        patch_size_meters_climate=patch_size_meters_climate,
                        patch_size_meters_terrain=patch_size_meters_terrain
                        )

if __name__ == "__main__":
    #variables = f'_L{use_landsat}_C{use_climate}_T{use_terrain}_'
    cloud_storage_cnn = False
    model_output_cnn = r'Model/CNN_Models/CNN_Latest.keras'
    model_output_rf = f'Model/RF_Models/RF_latest'
    
    rf = RF()
    cnn = CNN(use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage_cnn)

    '''Training Data'''
    print(f'\n Fetching training dataset for years {years}:\n')
    landsat_data, climate_data, terrain_data, targets = get_training_dataset()
    
    '''KerasTuner'''
    #keras_tuner()

    '''Pearsons Coefficient'''
    data_analysis.print_pearson_coefficient(input_data=landsat_data, target_data=targets)
    data_analysis.print_pearson_coefficient(input_data=climate_data, target_data=targets)
    data_analysis.print_pearson_coefficient(input_data=terrain_data, target_data=targets)

    '''RF'''
    train(model=rf, model_output_path=model_output_rf, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)

    '''CNN'''
    train(model=cnn, model_output_path=model_output_cnn, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)

    #cnn_model = CNN(model_path=model_output_cnn)
   
    '''Maps'''
    #plot_maps(model_kind='CNN', model_path=model_output_cnn, cloud_storage=cloud_storage_cnn)
  
    #save_training_patches_as_images()