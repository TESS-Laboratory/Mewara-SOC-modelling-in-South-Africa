import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_model_utils import base_model_utils
from Model.training_data_utils import training_data_utils
from Maps.maps_utils import map_utils

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

years = [1998, 1999, 2000, 2002, 2007, 2008, 2009, 2010, 2016, 2017, 2018]
#years = [1998, 2007, 2008, 2018]
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
    landsat_data, climate_data, terrain_data, targets = training_data_utils.get_training_data(
        soc_data_path=training_soc_path,
        years=years,
        start_month=start_month,
        end_month=end_month,
        patch_size_meters_landsat=patch_size_meters_landsat,
        patch_size_meters_climate=patch_size_meters_climate,
        patch_size_meters_terrain=patch_size_meters_terrain
        )
    return np.round(landsat_data, 1), np.round(climate_data, 1), np.round(terrain_data, 1), np.round(targets, 1)

def save_training_patches_as_images():
    years = [2008]
    soc_data = pd.read_csv(training_soc_path)
    lat_field = 'Lat'
    lon_field = 'Lon'
    terrain_patches_dict, landsat_patches_dict, climate_patches_dict = training_data_utils.get_patches(soc_data_path=r'DataProcessing\soc_gdf.csv',
                                                          folder_name='Train',
                                                          years=years, 
                                                          start_month=1,
                                                          end_month=12,
                                                          patch_size_meters_landsat=patch_size_meters_landsat,
                                                          patch_size_meters_climate=patch_size_meters_climate,
                                                          patch_size_meters_terrain=patch_size_meters_terrain,
                                                          lat_field='Lat',
                                                          lon_field='Lon')
    
    for year in years:
        for month in range(start_month, end_month + 1):
            soc_data_monthly = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month)]
                
            if soc_data_monthly.empty == True:
                continue

            lat_lon_pairs = list(zip(soc_data_monthly[lat_field], soc_data_monthly[lon_field]))

            for lat, lon in lat_lon_pairs:
                c_percent = soc_data_monthly[(soc_data_monthly[lat_field]==lat) & (soc_data_monthly[lon_field]==lon)]['C'].values[0]

                output_path_terrain_patch = f'Data/Patches/{year}/{month}_{lat}_{lon}_{c_percent}_terrain'
                terrain_patch = terrain_patches_dict.get((lat, lon))
                save_patch(terrain_patch, output_path_terrain_patch, 2)

                output_path_landsat_patch = f'Data/Patches/{year}/{month}_{lat}_{lon}_{c_percent}_landsat'
                landsat_patch = landsat_patches_dict.get((year, lat, lon))
                save_patch(landsat_patch, output_path_landsat_patch, 4)
            
                output_path_climate_patch = f'Data/Patches/{year}/{month}_{lat}_{lon}_{c_percent}_climate'
                climate_patch = climate_patches_dict.get((year, month, lat, lon))
                save_patch(climate_patch, output_path_climate_patch, 3)
               
def save_patch(patch, output_path, channels):
    for j in range(channels):
        img = patch[:, :, j]

        img_path = f'{output_path}_{j+1}.png'
        if not os.path.exists(img_path):
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{os.path.basename(img_path)}')
        plt.savefig(img_path)
        plt.close()             

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
    keras_tuner()

    '''CNN'''
    #train(model=cnn, model_output_path=model_output_cnn)

    '''RF'''
    #train(model=rf, model_output_path=model_output_rf)

    #cnn_model = CNN(model_path=model_output_cnn)
   
    '''Maps'''
    #plot_maps(model_kind='RF', model_path=model_output_rf)
  
    #save_training_patches_as_images()