import os
import numpy as np
from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_data_utils import base_data_utils
from Maps.maps_utils import map_utils
from Model.base_model import base_model
from Model.data_analysis import data_analysis

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

years = [1998, 1999, 2000, 2001, 2002, 2004, 2007, 2008, 2009, 2010, 2012, 2017, 2018]
#years = [2008, 2009]
start_month = 1
end_month = 12
epochs = 30

use_landsat = True
use_climate = True
use_terrain = True
use_cache = True
update_cache = True

patch_size_meters_landsat = 7680 # roughly 256*256 pixels
patch_size_meters_climate = 20000 # roughly 4*4 pixels
patch_size_meters_terrain = 960 # roughly 8*8 pixels

training_soc_path = r'DataProcessing/soc_gdf.csv'
landsat_bands = [0,4,5,6,7]
climate_bands = [0]
terrain_bands = [2]
training_kfold = 1

def get_training_dataset(landsat_bands, climate_bands, terrain_bands):
    lat_lon_data, landsat_data, climate_data, terrain_data, targets = base_data_utils.get_test_train_data(
        soc_data_path=training_soc_path,
        years=years,
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain,
        use_cache=use_cache,
        update_cache=update_cache) 
    
    return np.stack(lat_lon_data), (np.round(landsat_data, 1))[:,:,:,landsat_bands], (np.round(climate_data, 1))[:, :, :, climate_bands], \
(np.round(terrain_data, 1))[:, : ,:, terrain_bands], np.round(targets, 1)

def train(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
def train(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
    print(f'\n Training {model.__class__.__name__} model:\n')
    base_model.train_val_test_spatial_split(
                model=model,
                lat_lon_data=lat_lon_data,
                landsat_data=landsat_data,
                climate_data=climate_data,
                terrain_data=terrain_data,
                targets=targets,
                model_output_path=model_output_path,
                epochs=epochs,
                k_fold=training_kfold)

def keras_tuner(landsat_data, climate_data, terrain_data, targets, epochs):
    print('\n Tuning CNN Model: \n')
    KerasTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets, epochs=epochs)
    KerasTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets, epochs=epochs)

def get_model(model_kind, model_path, cloud_storage):
    if model_kind == 'RF':
        return RF(model_path=model_path, cloud_storage=cloud_storage)
    elif model_kind == 'CNN':
        return CNN(model_path=model_path, use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage)
    
def plot_maps(model_kind, model_path, cloud_storage):
    model = get_model(model_kind=model_kind, model_path=model_path, cloud_storage=cloud_storage)
    for year in years:
        map_utils.create_map(year=year, 
                        start_month=1, 
                        end_month=12,
                        model=model,
                        patch_size_meters_landsat=patch_size_meters_landsat,
                        patch_size_meters_climate=patch_size_meters_climate,
                        patch_size_meters_terrain=patch_size_meters_terrain,
                        landsat_bands=landsat_bands,
                        climate_bands=climate_bands,
                        terrain_bands=terrain_bands
                        )

if __name__ == "__main__":
    cloud_storage = False
    model_output_cnn = r'Model/CNN_Models/CNN_Model.keras'
    model_output_rf = r'Model/RF_Models/RF_Model'
    
    rf = RF(cloud_storage=cloud_storage)
    cnn = CNN(use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage)

    '''Pearsons Coefficient'''
    #lat_lon_data, landsat_data, climate_data, terrain_data, targets = get_training_dataset(landsat_bands=[0,1,2,3,4,5,6,7], climate_bands=[0,1,2], terrain_bands=[0,1,2,3])
    #data_analysis.plot_pearson_coefficient(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)

    '''Training Data'''
    lat_lon_data, landsat_data, climate_data, terrain_data, targets = get_training_dataset(landsat_bands=landsat_bands, climate_bands=climate_bands, terrain_bands=terrain_bands)

    '''KerasTuner'''
    keras_tuner(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets, epochs=epochs)

    '''CNN'''
    #train(model=cnn, model_output_path=model_output_cnn, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    
    '''RF'''
    #train(model=rf, model_output_path=model_output_rf, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)

    '''Maps'''
    #plot_maps(model_kind='CNN', model_path=model_output_cnn, cloud_storage=cloud_storage)
    
    #plot_maps(model_kind='RF', model_path=model_output_rf, cloud_storage=cloud_storage)

    #cnn_model = CNN(model_path=model_output_cnn)

    #save_training_patches_as_images()