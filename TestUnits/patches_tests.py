import pandas as pd
from Model.training_data_utils import training_data_utils
import rasterio

patch_size_meters_landsat = 15360 # roughly 16*16 pixels
patch_size_meters_climate = 15360 # roughly 4*4 pixels
patch_size_meters_terrain = 15360 # roughly 16*16 pixels
save_patches = True
year = 2018

def save_patches_year_months():
    soc_df = pd.read_csv('DataProcessing/soc_hex_grid.csv')
    soc_df = soc_df[soc_df['Year'] == year]

    for month in range(1,13):
        output_dir_low_carbon = f'TestUnits/Patches_{patch_size_meters_landsat}/{year}/Low_carbon/{month}'
        output_dir_high_carbon = f'TestUnits/Patches_{patch_size_meters_landsat}/{year}/High_carbon/{month}'
        soc_monthly = soc_df[soc_df['Month'] == month]
        low_carbon = soc_monthly[soc_monthly['C'] < 0.5]
        high_carbon = soc_monthly[soc_monthly['C'] > 5]

        low_carbon_lat_lons = list(set(zip(low_carbon['Lat'], low_carbon['Lon'])))
        high_carbon_lat_lons = list(set(zip(high_carbon['Lat'], high_carbon['Lon'])))

        test_landsat_patch(output_dir=output_dir_low_carbon, year=year, lat_lon_pairs=low_carbon_lat_lons)
        test_climate_patch(output_dir=output_dir_low_carbon, year=year, month=month, lat_lon_pairs=low_carbon_lat_lons)
        test_terrain_patch(output_dir=output_dir_low_carbon, lat_lon_pairs=low_carbon_lat_lons)

        test_landsat_patch(output_dir=output_dir_high_carbon, year=year, lat_lon_pairs=high_carbon_lat_lons)
        test_climate_patch(output_dir=output_dir_high_carbon, year=year, month=month, lat_lon_pairs=high_carbon_lat_lons)
        test_terrain_patch(output_dir=output_dir_high_carbon, lat_lon_pairs=high_carbon_lat_lons)
    
def test_landsat_patch(output_dir, year, lat_lon_pairs):
    raster_path=f'Data/LandSat/Annual_Processed/Landsat_{year}.tif'
    output_folder=f'{output_dir}/Landsat_{year}'

    # Landsat was downsampled by 4, therefore each pixel is 30 m * 4 = 120 meters
    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters_landsat, meters_per_pixel=120)

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
                print(f"Landsat patch for {lat}, {lon} missing")

def test_climate_patch(output_dir, year, month, lat_lon_pairs):  
    prec_raster_path =f'Data/WorldClim_SA/wc2.1_2.5m_prec_{year}-{month:02d}.tif'
    tmin_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmin_{year}-{month:02d}.tif'
    tmax_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmax_{year}-{month:02d}.tif'
       
    output_prec_folder=f'{output_dir}/Climate_{year}_{month}/Prec'
    output_tmin_folder=f'{output_dir}/Climate_{year}_{month}/Tmin'
    output_tmax_folder=f'{output_dir}/Climate_{year}_{month}/Tmax'

    meters_per_pixel = (2.5/60) * 111111 # 2.5 degrees from the equator ~ 111111 * 2.5 meters = 4,629.625 meters
    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters_climate, meters_per_pixel=meters_per_pixel)
    
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
                    print("Climate prec patch for {lat}, {lon} and month {month} missing")
                tmin_patch = training_data_utils.extract_patch(tmin_dataset, 
                                                               lat, 
                                                               lon, 
                                                               patch_size_pixels,
                                                               save_patch=save_patches, 
                                                               output_patch_folder=output_tmin_folder, 
                                                               output_patch_filename=output_filename)
                if tmin_patch is None or not tmin_patch.any():
                    print("Climate tmin patch for {lat}, {lon} and month {month} missing")
                tmax_patch = training_data_utils.extract_patch(tmax_dataset, 
                                                               lat, 
                                                               lon, 
                                                               patch_size_pixels,
                                                               save_patch=save_patches, 
                                                               output_patch_folder=output_tmax_folder, 
                                                               output_patch_filename=output_filename)
                if tmax_patch is None or not tmax_patch.any():
                    print("Climate tmax patch for {lat}, {lon} and month {month} missing")
   
def test_terrain_patch(output_dir, lat_lon_pairs):
    dem_path =f'Data/TerrainData/DEM.tif'
    aspect_path = f'Data/TerrainData/Aspect.tif'
    slope_path = f'Data/TerrainData/Slope.tif'
    twi_path = f'Data/TerrainData/TWI.tif'

    output_dem_folder=f'{output_dir}/Terrain/DEM'
    output_aspect_folder=f'{output_dir}/Terrain/Aspect'
    output_slope_folder=f'{output_dir}/Terrain/Slope'
    output_twi_folder=f'{output_dir}/Terrain/TWI'

    patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters_terrain, meters_per_pixel=120)
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

save_patches_year_months()