import os
import numpy as np
from MapsTrends.maps_utils import map_utils
from MapsTrends.trends_analysis import trends_analysis
from Model import GridSearchTuner
from Model.CNN import CNN
from Model.KerasTuner import KerasTuner
from Model.RF import RF
from Model.base_data_utils import base_data_utils
from Model.base_model import base_model
from Model.data_analysis import data_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2012, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
start_month = 1
end_month = 12
epochs = 30

use_landsat = True
use_climate = True
use_terrain = True
use_cache = True
update_cache = True

patch_size_meters_landsat = 30720 # roughly 128*128 pixels
patch_size_meters_climate = 30720 # roughly 4*4 pixels
patch_size_meters_terrain = 30720 # roughly 128*128 pixels

training_soc_path = r'DataProcessing/soc_hex_grid.csv'
landsat_bands = [4,5,6,7]
climate_bands = [0]
terrain_bands = [0,1,2]
training_kfold = 1
performance_metric_kfold = 10

def targets_density_plot(targets):
    # Create the density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(targets, bw_adjust=0.5, fill=True)
    plt.xlabel('Carbon (% by Mass)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title('Density Plot of Carbon Values', fontsize=18)
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 10.5, 0.5))
    plt.grid(True)
    plt.show()

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
    
    return lat_lon_data, np.round(landsat_data[:,:,:,landsat_bands],3), np.round(climate_data[:,:,:,climate_bands],3), np.round(terrain_data[:,:,:,terrain_bands],3), np.round(targets, 1)

def train(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
    print(f'\n Training {model.__class__.__name__} model:\n')
    base_model.train_val_test_spatial_split(
                model_class=model,
                lat_lon_data=lat_lon_data,
                landsat_data=landsat_data,
                climate_data=climate_data,
                terrain_data=terrain_data,
                targets=targets,
                model_output_path=model_output_path,
                epochs=epochs,
                k_fold=training_kfold)

def test(model_kind, model_path, cloud_storage, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
    model = get_model(model_kind=model_kind, model_path=model_path, cloud_storage=cloud_storage)
    print(f'\n Testing {model.__class__.__name__} model:\n')

    base_model.test(model=model,
                    lat_lon_data=lat_lon_data,
                    landsat_data=landsat_data,
                    climate_data=climate_data,
                    terrain_data=terrain_data,
                    targets=targets,
                    model_predictions_output_path=f'{os.path.dirname(model_path)}/Predictions/predictions_scatter.png')
    print(f'Predictions scatter plot saved')

def performance_metric(model, model_output_path, lat_lon_data, landsat_data, climate_data, terrain_data, targets):
    print(f'\n Training {model.__class__.__name__} model:\n')
    base_model.train_spatial_leave_cluster_out_split(
                model=model,
                lat_lon_data=lat_lon_data,
                landsat_data=landsat_data,
                climate_data=climate_data,
                terrain_data=terrain_data,
                targets=targets,
                model_output_path=model_output_path,
                epochs=epochs,
                k_fold=performance_metric_kfold)

def keras_tuner(landsat_data, climate_data, terrain_data, targets, epochs):
    print('\n Tuning CNN Model: \n')
    KerasTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets, epochs=50)

def get_model(model_kind, model_path, cloud_storage):
    if model_kind == 'RF':
        return RF(model_path=model_path, cloud_storage=cloud_storage)
    elif model_kind == 'CNN':
        return CNN(model_path=model_path, use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage)
    
def plot_maps(model_kind, model_path, cloud_storage):
    model = get_model(model_kind=model_kind, model_path=model_path, cloud_storage=cloud_storage)
    for year in years:
        map_utils.create_map(year=year, 
                        start_month=start_month, 
                        end_month=end_month,
                        model=model,
                        patch_size_meters_landsat=patch_size_meters_landsat,
                        patch_size_meters_climate=patch_size_meters_climate,
                        patch_size_meters_terrain=patch_size_meters_terrain,
                        landsat_bands=landsat_bands,
                        climate_bands=climate_bands,
                        terrain_bands=terrain_bands)

if __name__ == "__main__":
    cloud_storage = False
    model_output_cnn = f'Model/CNN_Models_{patch_size_meters_landsat}/CNN_Model.keras'
    model_output_rf = f'Model/RF_Models_{patch_size_meters_landsat}/RF_Model'
    
    rf = RF(cloud_storage=cloud_storage)
    cnn = CNN(use_landsat=use_landsat, use_climate=use_climate, use_terrain=use_terrain, cloud_storage=cloud_storage)

    '''Pearsons Coefficient'''
    #lat_lon_data, landsat_data, climate_data, terrain_data, targets = get_training_dataset(landsat_bands=[0,1,2,3,4,5,6,7], climate_bands=[0,1,2], terrain_bands=[0,1,2,3])
    #data_analysis.plot_pearson_coefficient(lat_lon_data = lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)

    '''Training Data'''
    lat_lon_data, landsat_data, climate_data, terrain_data, targets = get_training_dataset(landsat_bands=landsat_bands, climate_bands=climate_bands, terrain_bands=terrain_bands)
    #targets_density_plot(targets=targets)

    '''KerasTuner for CNN'''
    #keras_tuner(landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets, epochs=epochs)

    '''GridSearch for RF'''
    #GridSearchTuner.GridSearchTuner.search(input_landsat_data=landsat_data, input_climate_data=climate_data, input_terrain_data=terrain_data, targets=targets)

    '''RF'''
    #performance_metric(model=rf, model_output_path=model_output_rf, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    #train(model=rf, model_output_path=model_output_rf, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    #test(model_kind='RF', model_path=model_output_rf, cloud_storage=cloud_storage, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    #plot_maps(model_kind='RF', model_path=model_output_rf, cloud_storage=cloud_storage)

    '''CNN'''
    #performance_metric(model=cnn, model_output_path=model_output_cnn, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    train(model=cnn, model_output_path=model_output_cnn, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    test(model_kind='CNN', model_path=model_output_cnn, cloud_storage=cloud_storage, lat_lon_data=lat_lon_data, landsat_data=landsat_data, climate_data=climate_data, terrain_data=terrain_data, targets=targets)
    plot_maps(model_kind='CNN', model_path=model_output_cnn, cloud_storage=cloud_storage)
    

