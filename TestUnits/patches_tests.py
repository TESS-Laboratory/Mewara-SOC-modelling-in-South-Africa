from Model.training_data_utils import training_data_utils
import rasterio

patch_size_meters = 30720 # roughly 256*256 pixels for landsat and terrain, 7*7 pixels for climate

lat_lon_pairs = list(set([(-28.75085597205501, 29.858),  
                          #(-29.357073754704118, 31.258000000000003), 
                          #(-30.64777778, 29.46944444), 
                          #(-33.61111111, 18.46583333), 
                          #(-28.29477778, 26.48794444),
                          #(-32.07580556, 18.82205556),
                          #(-25.86833333, 28.05861111),
                          (-28.75085597, 29.058)]))

save_patches = True
year = 2008
month = 1

def test_terrain_patch():
    dem_path =f'Data\TerrainData\DEM.tif'
    aspect_path = f'Data\TerrainData\Aspect.tif'
    slope_path = f'Data\TerrainData\Slope.tif'
    twi_path = f'Data\TerrainData\TWI.tif'

    output_dem_folder=f'TestUnits/Patches/{patch_size_meters}/Terrain/DEM'
    output_aspect_folder=f'TestUnits/Patches/{patch_size_meters}/Terrain/Aspect'
    output_slope_folder=f'TestUnits/Patches/{patch_size_meters}/Terrain/Slope'
    output_twi_folder=f'TestUnits/Patches/{patch_size_meters}/Terrain/TWI'

    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)
    patch_size_pixels = 1
    
    with rasterio.open(dem_path) as dem_dataset, \
         rasterio.open(aspect_path) as aspect_dataset, \
         rasterio.open(slope_path) as slope_dataset, \
         rasterio.open(twi_path) as twi_dataset:

        for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                dem_patch = training_data_utils.extract_patch(dataset=dem_dataset, 
                                                                 lat=lat, 
                                                                 lon=lon, 
                                                                 patch_size_pixels=patch_size_pixels, 
                                                                 save_patch=save_patches, 
                                                                 output_patch_folder=output_dem_folder, 
                                                                 output_patch_filename=output_filename)
                if dem_patch is None or len(dem_patch) == 0:
                    print(f"DEM patch for {lat}, {lon} missing")
                
                slope_patch = training_data_utils.extract_patch(dataset=slope_dataset, 
                                                                 lat=lat, 
                                                                 lon=lon, 
                                                                 patch_size_pixels=patch_size_pixels, 
                                                                 save_patch=save_patches, 
                                                                 output_patch_folder=output_slope_folder, 
                                                                 output_patch_filename=output_filename)
                if slope_patch is None or len(slope_patch) == 0:
                    print(f"Slope patch for {lat}, {lon} missing")
                
                aspect_patch = training_data_utils.extract_patch(dataset=aspect_dataset, 
                                                                 lat=lat, 
                                                                 lon=lon, 
                                                                 patch_size_pixels=patch_size_pixels, 
                                                                 save_patch=save_patches, 
                                                                 output_patch_folder=output_aspect_folder, 
                                                                 output_patch_filename=output_filename)
                if aspect_patch is None or len(aspect_patch) == 0:
                    print(f"Aspect patch for {lat}, {lon} missing")

                twi_patch = training_data_utils.extract_patch(dataset=twi_dataset, 
                                                              lat=lat, 
                                                              lon=lon, 
                                                              patch_size_pixels=patch_size_pixels,
                                                              save_patch=save_patches, 
                                                              output_patch_folder=output_twi_folder, 
                                                              output_patch_filename=output_filename)
                if twi_patch is None or len(twi_patch) == 0:
                    print(f"TWI patch for {lat}, {lon} missing")

def test_climate_patch():
    prec_raster_path =f'Data/WorldClim_SA/wc2.1_2.5m_prec_{year}-{month:02d}.tif'
    tmin_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmin_{year}-{month:02d}.tif'
    tmax_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmax_{year}-{month:02d}.tif'
       
    output_prec_folder=f'TestUnits/Patches/{patch_size_meters}/Climate_{year}_{month}/Prec'
    output_tmin_folder=f'TestUnits/Patches/{patch_size_meters}/Climate_{year}_{month}/Tmin'
    output_tmax_folder=f'TestUnits/Patches/{patch_size_meters}/Climate_{year}_{month}/Tmax'

    meters_per_pixel = (2.5/60) * 111111 # 2.5 degrees from the equator ~ 111111 * 2.5 meters = 4,629.625 meters
    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=meters_per_pixel)
    
    with rasterio.open(prec_raster_path) as prec_dataset, \
             rasterio.open(tmin_raster_path) as tmin_dataset, \
             rasterio.open(tmax_raster_path) as tmax_dataset: \
            
            for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                prec_patch = training_data_utils.extract_patch(dataset=prec_dataset, 
                                                    lat=lat, 
                                                    lon=lon, 
                                                    patch_size_pixels=patch_size_pixels,  
                                                    save_patch=save_patches, 
                                                    output_patch_folder=output_prec_folder, 
                                                    output_patch_filename=output_filename)
                if prec_patch is None or not prec_patch.any():
                    raise ValueError('Climate prec patch missing')
                tmin_patch = training_data_utils.extract_patch(tmin_dataset, 
                                                               lat, 
                                                               lon, 
                                                               patch_size_pixels,
                                                               save_patch=save_patches, 
                                                               output_patch_folder=output_tmin_folder, 
                                                               output_patch_filename=output_filename)
                if tmin_patch is None or not tmin_patch.any():
                    raise ValueError('Climate tmin patch missing')
                tmax_patch = training_data_utils.extract_patch(tmax_dataset, 
                                                               lat, 
                                                               lon, 
                                                               patch_size_pixels,
                                                               save_patch=save_patches, 
                                                               output_patch_folder=output_tmax_folder, 
                                                               output_patch_filename=output_filename)
                if tmax_patch is None or not tmax_patch.any():
                    raise ValueError('Climate tmax patch missing')

def test_landsat_patch():
    raster_path=f'Data/LandSat/Annual_Processed/Landsat_{year}.tif'
    output_folder=f'TestUnits/Patches/{patch_size_meters}/Landsat_{year}'

    # Landsat was downsampled by 4, therefore each pixel is 30 m * 4 = 120 meters
    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)

    with rasterio.open(raster_path) as dataset:
        for lat, lon in lat_lon_pairs:
            output_filename=f'({lat}_{lon}).tif'
            patch = training_data_utils.extract_patch(dataset=dataset, 
                                                lat=lat, 
                                                lon=lon, 
                                                patch_size_pixels=patch_size_pixels,  
                                                save_patch=save_patches, 
                                                output_patch_folder=output_folder, 
                                                output_patch_filename=output_filename)
            if patch is None or not patch.any():
                raise ValueError(f"Landsat patch for {lat}, {lon} missing")
                 
test_terrain_patch()
test_climate_patch()
test_landsat_patch()