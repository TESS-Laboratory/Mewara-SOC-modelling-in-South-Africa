from Model.CNN import CNN
from Model.RF import RF
from Model.data_utils import data_utils

def test(lat_lon_pairs, start_year, end_year, start_month, end_month, patch_size, model):
        print(f'\n Testing {model.__class__.__name__}:\n')
        for idx in range(len(lat_lon_pairs)):
            lat_lon = [lat_lon_pairs[idx]]
            print(f'\n C % for Lat Lon ({lat_lon}):\n')
            for year in range(start_year, end_year + 1):
                for month in range(start_month, end_month + 1):
                    climate_data_monthly = data_utils.get_monthly_climate_data(year=year, month=month, lat_lon_pairs=lat_lon, patch_size=patch_size)
                    landsat_data_monthly = data_utils.get_landsat_data(year=year, lat_lon_pairs=lat_lon, patch_size=patch_size)
                    terrain_data_monthly = data_utils.get_terrain_data(lat_lon_pairs=lat_lon, patch_size=patch_size)

                    predictions = model.predict(landsat_data_monthly, climate_data_monthly, terrain_data_monthly)
                    print(f'\t{lat_lon}; Year_Month: {year}_{month} = {predictions[0]}\n')

if __name__ == "__main__":
    patch_size = 16
    epochs = 10
    landsat_data, climate_data, terrain_data, targets = data_utils.get_training_data(soc_data_path=r'DataPreprocessing\soc_gdf.csv',
                                                                                     start_year=2008,
                                                                                     end_year=2008,
                                                                                     start_month=1,
                                                                                     end_month=3,
                                                                                     patch_size=patch_size)
    cnn = CNN()
    print(f'\n Training CNN model:\n')
    cnn.train(landsat_data=landsat_data,
             climate_data=climate_data,
             terrain_data=terrain_data,
             targets=targets,
             model_output_path=f'Model\Soil_Carbon_CNN_{epochs}.keras',
             epochs=epochs)
    
    rf = RF()
    print(f'\n Training RF model:\n')
    rf.train(landsat_data=landsat_data,
             climate_data=climate_data,
             terrain_data=terrain_data,
             targets=targets)
    
    # C% for (-25.8065,27.84747222) = 2.43
    # C% for (-30.30497222, 29.54316667) = 4.27
    lat_lon_pairs = [(-25.8065,27.84747222), (-30.30497222, 29.54316667)]
    
    test(lat_lon_pairs=lat_lon_pairs, start_year=2008, end_year=2008, start_month=1, end_month=2, patch_size=patch_size, model=cnn)
    test(lat_lon_pairs=lat_lon_pairs, start_year=2008, end_year=2008, start_month=1, end_month=2, patch_size=patch_size, model=rf)
        