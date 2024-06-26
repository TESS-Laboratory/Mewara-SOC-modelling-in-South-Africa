import calendar
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

class training_data_utils:
    @staticmethod
    def lat_lon_to_pixel(dataset, lat, lon):
        transform = dataset.transform
        pixel_x, pixel_y = ~transform * (lon, lat)
        return int(pixel_x), int(pixel_y)

    @staticmethod
    def get_patch_size_pixels(patch_size_meters, meters_per_pixel):
        return round(patch_size_meters / meters_per_pixel)

    @staticmethod
    def extract_patch(dataset, lat, lon, patch_size_pixels, save_patch = False, output_patch_folder = '', output_patch_filename = ''):
        # South Africa's geographical boundaries
        south_africa_bounds = {
            'min_lat': -35,
            'max_lat': -22,
            'min_lon': 16,
            'max_lon': 33
        }

        lat = float(lat)
        lon = float(lon)

        # Check if the coordinates are within South Africa's boundary
        if not (south_africa_bounds['min_lat'] <= lat <= south_africa_bounds['max_lat'] and
                south_africa_bounds['min_lon'] <= lon <= south_africa_bounds['max_lon']):
            return None
        
        pixel_x, pixel_y = training_data_utils.lat_lon_to_pixel(dataset, lat, lon)
        half_patch = patch_size_pixels // 2

        window_col_off = max(pixel_x - half_patch, 0)
        window_row_off = max(pixel_y - half_patch, 0)
        window_width = min(patch_size_pixels, dataset.width - window_col_off)
        window_height = min(patch_size_pixels, dataset.height - window_row_off)
        
        window = Window(window_col_off, window_row_off, window_width, window_height)

        # Read the data within the window
        patch = dataset.read(window=window)

        data_augment = False
        save_patch = False

        if (data_augment) :
            patch_size = patch_size_pixels

            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                # Pad the patch with np.nan values
                pad_width = ((0, 0), 
                            (0, patch_size - patch.shape[1]), 
                            (0, patch_size - patch.shape[2]))
                patch = np.pad(patch, pad_width, mode='constant', constant_values=np.nan)

        patch = training_data_utils.replace_nan_inf_data(data=patch)

        if (save_patch):
            res = (dataset.transform.a, dataset.transform.e)
            lon_min, lat_max = dataset.xy(window_row_off, window_col_off)

            transform = from_origin(
                lon_min,
                lat_max,
                res[0], 
                abs(res[1])
             )
            
            training_data_utils.save_patch_as_raster(patch=patch, 
                                            patch_size=patch_size if data_augment else patch_size_pixels, 
                                            is_augmented=data_augment,
                                            transform=transform,
                                            dataset=dataset,
                                            output_folder=output_patch_folder,
                                            output_filename=output_patch_filename)

        return patch
        
    def save_patch_as_raster(patch, patch_size, is_augmented, transform, dataset, output_folder, output_filename):
        if is_augmented:
            output_folder = f'{output_folder}\Augmented'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_path = os.path.join(output_folder, output_filename)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=patch_size,
            width=patch_size,
            count=patch.shape[0],  # Number of bands
            dtype=patch.dtype,
            crs=dataset.crs,
            transform=transform,
        ) as dst:
            for band in range(patch.shape[0]):
                dst.write(patch[band], band + 1)
    
    def replace_nan_inf_data(data):
        # Replace Inf with the maximum finite value in each slice
        max_value = np.nanmax(np.where(np.isfinite(data), data, -np.inf), axis=(0, 1))
        mask_inf = np.isinf(data)
        for i in range(data.shape[-1]):
            data[mask_inf[..., i], i] = max_value[i]

        # Calculate mean values ignoring NaN
        mean_value = np.empty(data.shape[-1])
        for i in range(data.shape[-1]):
            slice_data = data[..., i]
            if np.all(np.isnan(slice_data)):
                mean_value[i] = np.nan  # Default value when all elements are NaN
            else:
                mean_value[i] = np.nanmean(slice_data)
        
        # Replace NaN with mean values
        mask_nan = np.isnan(data)
        for i in range(data.shape[-1]):
            data[mask_nan[..., i], i] = mean_value[i]

        # Identify rows that are completely invalid (all NaN or Inf)
        invalid_rows = np.all(np.logical_or(np.isnan(data), np.isinf(data)), axis=(1, 2))

        return data[~invalid_rows]
   
    def get_terrain_patches(lat_lon_pairs, patch_size_meters):
        dem_path =f'Data\TerrainData\Elevation\DEM.tif'
        slope_path = f'Data\TerrainData\Elevation\Slope.tif'
        twi_path = f'Data\TerrainData\Elevation\TWI.tif'

        output_dem_folder=f'DataProcessing\Patches\Terrain\DEM\{patch_size_meters}'
        output_slope_folder=f'DataProcessing\Patches\Terrain\Slope\{patch_size_meters}'
        output_twi_folder=f'DataProcessing\Patches\Terrain\TWI\{patch_size_meters}'

        patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)
        
        if not (os.path.exists(dem_path) and os.path.exists(slope_path) and os.path.exists(twi_path)):
            return None
        
        terrain_patches = []

        with rasterio.open(dem_path) as dem_dataset, \
             rasterio.open(slope_path) as slope_dataset, \
             rasterio.open(twi_path) as twi_dataset:

            for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                dem_patch = training_data_utils.extract_patch(dataset=dem_dataset, lat=lat, lon=lon, patch_size_pixels=patch_size_pixels,
                                                    save_patch=True, 
                                                    output_patch_folder=output_dem_folder, 
                                                    output_patch_filename=output_filename)
                if dem_patch is None or not dem_patch.any():
                    terrain_patches.append(None)
                    continue
                slope_patch = training_data_utils.extract_patch(dataset=slope_dataset, lat=lat, lon=lon, patch_size_pixels=patch_size_pixels)
                if slope_patch is None or not slope_patch.any():
                    terrain_patches.append(None)
                    continue
                twi_patch = training_data_utils.extract_patch(dataset=twi_dataset, lat=lat, lon=lon, patch_size_pixels=patch_size_pixels)
                if twi_patch is None or not twi_patch.any():
                    terrain_patches.append(None)
                    continue
                stacked_patch = np.stack([dem_patch[0], slope_patch[0], twi_patch[0]], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                terrain_patches.append(stacked_patch)
        return terrain_patches
    
    def get_climate_patches(year, month, lat_lon_pairs, patch_size_meters):
        prec_raster_path =f'Data\WorldClim_SA\wc2.1_2.5m_prec_{year}-{month:02d}.tif'
        tmin_raster_path = f'Data\WorldClim_SA\wc2.1_2.5m_tmin_{year}-{month:02d}.tif'
        tmax_raster_path = f'Data\WorldClim_SA\wc2.1_2.5m_tmax_{year}-{month:02d}.tif'
       
        '''
        pixel_resolution = 2.5 arc minutes
        1 degree = 60 arc minutes
        2.5 minutes = 2.5/60 degrees
        1 degree = 111111 meters
        meters_per_pixel = 2.5/60 * 111111 = 4629.625 meters
        no_of_pixels = patch_size_meters / meters_per_pixel
        no_of_pixels * meters_per_pixel = patch_size_meters
        2 * 4629.625 ~ 10 Km
        '''
        meters_per_pixel = (2.5/60) * 111111 # 2.5 degrees from the equator ~ 111111 * 2.5 meters = 4,629.625 meters
        patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=meters_per_pixel)

        output_prec_patch_folder=f'DataProcessing\Patches\Climate\Precipitation\{patch_size_meters}\{year}\{month}'
       
        if not (os.path.exists(prec_raster_path) and os.path.exists(tmin_raster_path) and os.path.exists(tmax_raster_path)):
            return None
        
        _, no_of_days = calendar.monthrange(year=year, month=month)
        climate_patches = []

        with rasterio.open(prec_raster_path) as prec_dataset, \
             rasterio.open(tmin_raster_path) as tmin_dataset, \
             rasterio.open(tmax_raster_path) as tmax_dataset: \
            
            for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                prec_patch = training_data_utils.extract_patch(dataset=prec_dataset, 
                                                    lat=lat, 
                                                    lon=lon, 
                                                    patch_size_pixels=patch_size_pixels,  
                                                    save_patch=True, 
                                                    output_patch_folder=output_prec_patch_folder, 
                                                    output_patch_filename=output_filename)
                if prec_patch is None or not prec_patch.any():
                    climate_patches.append(None)
                    continue
                tmin_patch = training_data_utils.extract_patch(tmin_dataset, lat, lon, patch_size_pixels)
                if tmin_patch is None or not tmin_patch.any():
                    climate_patches.append(None)
                    continue
                tmax_patch = training_data_utils.extract_patch(tmax_dataset, lat, lon, patch_size_pixels)
                if tmax_patch is None or not tmax_patch.any():
                    climate_patches.append(None)
                    continue
                stacked_patch = np.stack([prec_patch[0]/no_of_days, tmin_patch[0], tmax_patch[0]], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                climate_patches.append(stacked_patch)  
        return climate_patches
    
    def get_landsat_patches(year, lat_lon_pairs, patch_size_meters):
        raster_path=f'Data\LandSat\Annual_Processed\{year}\Landsat_{year}.tif'
        output_folder=f'DataProcessing\Patches\Landsat\{patch_size_meters}\{year}'

        # Landsat was downsampled by 4, therefore each pixel is 30 m * 4 = 120 meters
        patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)

        if not os.path.exists(raster_path):
            return None
        
        landsat_patches = []
        
        with rasterio.open(raster_path) as dataset:
            for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                patch = training_data_utils.extract_patch(dataset=dataset, 
                                                lat=lat, 
                                                lon=lon, 
                                                patch_size_pixels=patch_size_pixels,  
                                                save_patch=True, 
                                                output_patch_folder=output_folder, 
                                                output_patch_filename=output_filename)
                if patch is None or not patch.any():
                    landsat_patches.append(None)
                    continue
                nir_band = patch[3]
                ndvi_band = patch[4]
                evi_band = patch[5]
                savi_band = patch[6]
                rvi_band = patch[7]
                stacked_patch = np.stack([ndvi_band, evi_band, rvi_band, savi_band], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                landsat_patches.append(stacked_patch)
        return landsat_patches
    
    def append_patch(year, month, lat, lon, c_percent, covariates, patch):
        num_channels = patch.shape[-1]
       
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                covariate = [year, month, lat, lon, c_percent]
                for k in range(num_channels):
                    covariate.append(patch[i, j, k])
                covariates.append(covariate)
                
    def get_training_data(soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain):
        soc_data = pd.read_csv(soc_data_path)
        covariate_columns_landsat = ['YEAR', 'MONTH', 'LAT', 'LON', 'C', 'NIR', 'NDVI', 'EVI', 'RVI']  
        covariate_columns_climate = ['YEAR', 'MONTH', 'LAT', 'LON', 'C', 'PREC', 'TMIN', 'TMAX']
        covariate_columns_terrain = ['YEAR', 'MONTH', 'LAT', 'LON', 'C', 'DEM', 'SLOPE', 'TWI'] 
        covariates_landsat = []
        covariates_climate = []
        covariates_terrain = []

        landsat_data = []
        climate_data = []
        terrain_data = []
        targets = []

        for year in years:
            for month in range(start_month, end_month + 1):
                soc_data_monthly = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month)]
                
                if soc_data_monthly.empty == True:
                    continue

                lat_lon_pairs_monthly = list(zip(soc_data_monthly['Lat'], soc_data_monthly['Lon']))

                landsat_patches = training_data_utils.get_landsat_patches(year=year, lat_lon_pairs=lat_lon_pairs_monthly, patch_size_meters=patch_size_meters_landsat)
                if landsat_patches is None:
                    continue

                climate_patches = training_data_utils.get_climate_patches(year=year, month=month, lat_lon_pairs=lat_lon_pairs_monthly, patch_size_meters=patch_size_meters_climate)
                if climate_patches is None:
                    continue

                terrain_patches = training_data_utils.get_terrain_patches(lat_lon_pairs=lat_lon_pairs_monthly, patch_size_meters=patch_size_meters_terrain)
                if terrain_patches is None:
                    continue

                for idx in range(len(lat_lon_pairs_monthly)):
                    lat, lon = lat_lon_pairs_monthly[idx]
                    c_percent = soc_data_monthly.iloc[idx]['C']
                    
                    landsat_patch = landsat_patches[idx]
                    if landsat_patch is None:
                        continue

                    climate_patch = climate_patches[idx]
                    if climate_patch is None:
                        continue

                    terrain_patch = terrain_patches[idx]
                    if terrain_patch is None:
                        continue

                    landsat_data.append(landsat_patch)
                    climate_data.append(climate_patch)
                    terrain_data.append(terrain_patch)
                    targets.append(c_percent)
                    '''
                    data_utils.append_patch(year=year, month=month, lat=lat, lon=lon, c_percent=c_percent, covariates=covariates_landsat, patch=landsat_patch)
                    data_utils.append_patch(year=year, month=month, lat=lat, lon=lon, c_percent=c_percent, covariates=covariates_climate, patch=climate_patch)
                    data_utils.append_patch(year=year, month=month, lat=lat, lon=lon, c_percent=c_percent, covariates=covariates_terrain, patch=terrain_patch)  
                    '''
            '''
            data_utils.save_csv(arr=covariates_landsat, column_names=covariate_columns_landsat, output_path = f'DataProcessing\Covariates\{year}\Landsat.csv')
            data_utils.save_csv(arr=covariates_climate, column_names=covariate_columns_climate, output_path = f'DataProcessing\Covariates\{year}\Climate.csv')
            data_utils.save_csv(arr=covariates_terrain, column_names=covariate_columns_terrain, output_path = f'DataProcessing\Covariates\{year}\Terrain.csv')
            '''
        return np.array(landsat_data), np.array(climate_data), np.array(terrain_data), np.array(targets)

    def save_csv(arr, column_names, output_path):    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        covariates_df = pd.DataFrame(arr, columns=column_names)
        covariates_df.to_csv(output_path, index=False)
        print(f'\nCovariates saved to {output_path}\n')