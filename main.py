from pydoc import cli
from tracemalloc import start
from Model.CNN import CNN
from Model.data_utils import data_utils

if __name__ == "__main__":
    patch_size = 8
    landsat_data, climate_data, terrain_data, targets = data_utils.get_training_data(soc_data_path=r'DataPreprocessing\soc_gdf.csv',
                                                                                     start_year=2008,
                                                                                     end_year=2008,
                                                                                     patch_size=patch_size)
    cnn = CNN()
    cnn._build_model(landsat_data=landsat_data,
                    climate_data=climate_data,
                    terrain_data=terrain_data)
    cnn.train(landsat_data=landsat_data,
             climate_data=climate_data,
             terrain_data=terrain_data,
             targets=targets,
             model_output_path=r'Model\Soil_Carbon_CNN_1.h5',
             epochs=1)
    
    lat_lon_pairs = [(-24.40593, 31.76698), (-24.40633, 31.76692)]
    landsat_test_data, climate_test_data, terrain_test_data = data_utils.get_test_data(lat_lon_pairs=lat_lon_pairs,
                                                                        start_year=2008,
                                                                        end_year=2008,
                                                                        patch_size=patch_size)
    predictions = cnn.predict(landsat_test_data, terrain_test_data, climate_test_data)
    print(predictions)