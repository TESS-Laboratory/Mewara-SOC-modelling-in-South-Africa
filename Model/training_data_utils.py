import calendar
import os
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from pyproj import Transformer

class training_data_utils:
    @staticmethod
    def lat_lon_to_pixel(dataset, lat, lon):
        if (dataset.crs.to_epsg() == 3857):
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
            x, y = transformer.transform(lat, lon)    
        elif (dataset.crs.to_epsg() == 4326):
            x, y = lon, lat
        else:
           raise ValueError("Unknown CRS")

        transform = dataset.transform
        pixel_x, pixel_y = ~transform * (x, y)
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

        data_augment = True
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

        return patch
        
    def replace_nan_inf_data(data):
       # Mask out NaNs, Infs, and -Infs
        mask_invalid = np.logical_or(np.isnan(data), np.isinf(data))
        data_valid = np.ma.masked_array(data, mask=mask_invalid)
        
        # Calculate mean values ignoring NaN, Inf, or -Inf
        mean_value = np.nanmean(data_valid, axis=(0, 1))
        
        # Replace NaN, Inf, and -Inf with mean values
        for i in range(data.shape[-1]):
            data[mask_invalid[..., i], i] = mean_value[i]

        # Identify rows that are completely invalid (all NaN or Inf or -Inf)
        invalid_rows = np.all(np.logical_or(np.isnan(data), np.isinf(data), np.isneginf(data)), axis=(1, 2))

        return data[~invalid_rows]
   
    def get_bd_patches_dict(lat_lon_pairs):
        raster_path=f'Data/BulkDensity/BulkDensity.tif'
      
        if not os.path.exists(raster_path):
            return None
        
        bd_patches_dict = {}
        
        with rasterio.open(raster_path) as dataset:
            for lat, lon in lat_lon_pairs:
                patch = training_data_utils.extract_patch(dataset=dataset, 
                                                lat=lat, 
                                                lon=lon, 
                                                patch_size_pixels=1)
                if patch is None or not patch.any():
                    bd_patches_dict[(lat, lon)] = None
                    continue
                bd_patches_dict[(lat, lon)] = patch[0]
        return bd_patches_dict
    
    def get_terrain_patches_dict(lat_lon_pairs, patch_size_meters):
        dem_path =f'Data/TerrainData/Elevation/DEM.tif'
        aspect_path = f'Data/TerrainData/Elevation/Aspect.tif'
        twi_path = f'Data/TerrainData/Elevation/TWI.tif'

        output_dem_folder=f'DataProcessing/Patches/Terrain/DEM/{patch_size_meters}'
        output_aspect_folder=f'DataProcessing/Patches/Terrain/Aspect/{patch_size_meters}'
        output_twi_folder=f'DataProcessing/Patches/Terrain/TWI/{patch_size_meters}'

        patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)
        
        if not (os.path.exists(dem_path) and os.path.exists(aspect_path) and os.path.exists(twi_path)):
            return None
        
        terrain_patches_dict = {}

        with rasterio.open(dem_path) as dem_dataset, \
             rasterio.open(aspect_path) as aspect_dataset, \
             rasterio.open(twi_path) as twi_dataset:

            for lat, lon in lat_lon_pairs:
                output_filename=f'({lat}_{lon}).tif'
                aspect_patch = training_data_utils.extract_patch(dataset=aspect_dataset, lat=lat, lon=lon, patch_size_pixels=patch_size_pixels)
                if aspect_patch is None or not aspect_patch.any():
                    terrain_patches_dict[(lat, lon)]= None
                    continue
                twi_patch = training_data_utils.extract_patch(dataset=twi_dataset, lat=lat, lon=lon, patch_size_pixels=patch_size_pixels)
                if twi_patch is None or not twi_patch.any():
                    terrain_patches_dict[(lat, lon)]= None
                    continue
                stacked_patch = np.stack([aspect_patch[0], twi_patch[0]], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                terrain_patches_dict[(lat, lon)] = stacked_patch
        return terrain_patches_dict
    
    def get_climate_patches_dict(year, month, lat_lon_pairs, patch_size_meters):
        prec_raster_path =f'Data/WorldClim_SA/wc2.1_2.5m_prec_{year}-{month:02d}.tif'
        tmin_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmin_{year}-{month:02d}.tif'
        tmax_raster_path = f'Data/WorldClim_SA/wc2.1_2.5m_tmax_{year}-{month:02d}.tif'
       
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

        output_prec_patch_folder=f'DataProcessing/Patches/Climate/Precipitation/{patch_size_meters}/{year}/{month}'
       
        if not (os.path.exists(prec_raster_path) and os.path.exists(tmin_raster_path) and os.path.exists(tmax_raster_path)):
            return None
        
        _, no_of_days = calendar.monthrange(year=year, month=month)
        climate_patches_dict = {}

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
                    climate_patches_dict[(year, month, lat, lon)]= None
                    continue
                tmin_patch = training_data_utils.extract_patch(tmin_dataset, lat, lon, patch_size_pixels)
                if tmin_patch is None or not tmin_patch.any():
                    climate_patches_dict[(year, month, lat, lon)]= None
                    continue
                tmax_patch = training_data_utils.extract_patch(tmax_dataset, lat, lon, patch_size_pixels)
                if tmax_patch is None or not tmax_patch.any():
                    climate_patches_dict[(year, month, lat, lon)]= None
                    continue
                stacked_patch = np.stack([prec_patch[0]/no_of_days, tmin_patch[0], tmax_patch[0]], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                climate_patches_dict[(year, month, lat, lon)] = stacked_patch
        return climate_patches_dict
    
    def get_landsat_patches_dict(year, lat_lon_pairs, patch_size_meters):
        raster_path=f'Data/LandSat/Annual_Processed/{year}/Landsat_{year}.tif'
        output_folder=f'DataProcessing/Patches/Landsat/{patch_size_meters}/{year}'

        # Landsat was downsampled by 4, therefore each pixel is 30 m * 4 = 120 meters
        patch_size_pixels = training_data_utils.get_patch_size_pixels(patch_size_meters=patch_size_meters, meters_per_pixel=120)

        if not os.path.exists(raster_path):
            return None
        
        landsat_patches_dict = {}
        
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
                    landsat_patches_dict[(year, lat, lon)] = None
                    continue
                red_band = patch[0]
                green_band = patch[1]
                blue_band = patch[2]
                nir_band = patch[3]
                ndvi_band = patch[4]
                evi_band = patch[5]
                savi_band = patch[6]
                rvi_band = patch[7]
                stacked_patch = np.stack([ndvi_band, evi_band, savi_band, rvi_band], axis=-1)
                stacked_patch = training_data_utils.replace_nan_inf_data(stacked_patch)
                landsat_patches_dict[(year, lat, lon)] = stacked_patch
        return landsat_patches_dict
     
    def get_training_data(soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain):
        return training_data_utils._get_training_test_data(prefix='Train',
                                                    soc_data_path=soc_data_path,
                                                    years=years,
                                                    start_month=start_month,
                                                    end_month=end_month,
                                                    patch_size_meters_landsat=patch_size_meters_landsat,
                                                    patch_size_meters_climate=patch_size_meters_climate,
                                                    patch_size_meters_terrain=patch_size_meters_terrain)
        
    def get_patches(soc_data, folder_name, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, lat_field, lon_field):
        lat_lon_pairs = list(set(zip(soc_data[lat_field], soc_data[lon_field])))
  
        print(f'\n Fetching {folder_name} data:\n')
        all_terrain_patches_dict = {}
        all_landsat_patches_dict = {}
        all_climate_patches_dict = {}

        terrain_patches_path = f"Data/{folder_name}/Terrain/{folder_name}_terrain.h5"
        if os.path.exists(terrain_patches_path):
            terrain_patches_dict = training_data_utils.load_patches(terrain_patches_path)
        else:
            terrain_patches_dict = training_data_utils.get_terrain_patches_dict(lat_lon_pairs=lat_lon_pairs, patch_size_meters=patch_size_meters_terrain)
            training_data_utils.save_patches(output_path=terrain_patches_path, patches_dict=terrain_patches_dict)

        if terrain_patches_dict is None:
            return None
        
        all_terrain_patches_dict.update(terrain_patches_dict)
        
        for year in years:
            print(f'\nProcessing {folder_name} {year}\n')
            soc_data_yearly = soc_data[(soc_data['Year'] == year)]
            lat_lon_pairs_yearly = list(set(zip(soc_data_yearly[lat_field], soc_data_yearly[lon_field])))
            
            landsat_patches_path = f"Data/{folder_name}/Landsat/{year}/{folder_name}_landsat_{year}.h5"
            if os.path.exists(landsat_patches_path):
                landsat_patches_dict = training_data_utils.load_patches(landsat_patches_path)
            else:
                landsat_patches_dict = training_data_utils.get_landsat_patches_dict(year=year, lat_lon_pairs=lat_lon_pairs_yearly, patch_size_meters=patch_size_meters_landsat)
                training_data_utils.save_patches(output_path=landsat_patches_path, patches_dict=landsat_patches_dict)

            if landsat_patches_dict is None:
                continue

            all_landsat_patches_dict.update(landsat_patches_dict)

            for month in range(start_month, end_month + 1):
                soc_data_monthly = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month)]
                
                if soc_data_monthly.empty == True:
                    continue

                lat_lon_pairs_monthly = list(set(zip(soc_data_monthly[lat_field], soc_data_monthly[lon_field])))

                climate_patches_path = f"Data/{folder_name}/Climate/{year}/{folder_name}_climate_{year}_{month}.h5"
                if os.path.exists(climate_patches_path):
                    climate_patches_dict = training_data_utils.load_patches(climate_patches_path)
                else:
                    climate_patches_dict = training_data_utils.get_climate_patches_dict(year=year, month=month, lat_lon_pairs=lat_lon_pairs_monthly, patch_size_meters=patch_size_meters_climate)
                    training_data_utils.save_patches(output_path=climate_patches_path, patches_dict=climate_patches_dict)

                if climate_patches_dict is None:
                    continue
                all_climate_patches_dict.update(climate_patches_dict)
        return all_terrain_patches_dict, all_landsat_patches_dict, all_climate_patches_dict
                
    def _get_training_test_data(prefix, soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, lat_field = 'Lat', lon_field = 'Lon'):
        soc_data = pd.read_csv(soc_data_path)

        landsat_data = []
        climate_data = []
        terrain_data = []
        targets = []

        print(f'\n Fetching {prefix} data:\n')

        terrain_patches_dict, landsat_patches_dict, climate_patches_dict = training_data_utils.get_patches(soc_data=soc_data,
                                                                                                            folder_name=prefix,
                                                                                                            years=years,
                                                                                                            start_month=start_month,
                                                                                                            end_month=end_month,
                                                                                                            patch_size_meters_terrain=patch_size_meters_terrain,
                                                                                                            patch_size_meters_landsat=patch_size_meters_landsat,
                                                                                                            patch_size_meters_climate=patch_size_meters_climate,
                                                                                                            lat_field=lat_field,
                                                                                                            lon_field=lon_field)
        for year in years:
            print(f'\nProcessing {prefix} {year}\n')
            
            for month in range(start_month, end_month + 1):
                soc_data_monthly = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month)]
                
                if soc_data_monthly.empty == True:
                    continue

                lat_lon_pairs_monthly = list(set(zip(soc_data_monthly[lat_field], soc_data_monthly[lon_field])))

                for lat, lon in lat_lon_pairs_monthly:
                    terrain_patch = terrain_patches_dict.get((lat, lon))
                    landsat_patch = landsat_patches_dict.get((year, lat, lon))
                    climate_patch = climate_patches_dict.get((year, month, lat, lon))

                    if terrain_patch is not None and landsat_patch is not None and climate_patch is not None:
                        c_percent = soc_data_monthly[(soc_data_monthly[lat_field] == lat) & (soc_data_monthly[lon_field] == lon)]['C'].values[0]

                        landsat_data.append(landsat_patch)
                        climate_data.append(climate_patch)
                        terrain_data.append(terrain_patch)
                        targets.append(c_percent)
                            
        return np.array(landsat_data), np.array(climate_data), np.array(terrain_data), np.array(targets)

    def save_patches(output_path, patches_dict):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with h5py.File(output_path, 'w') as h5f:
            for key, value in patches_dict.items():
                if len(key) == 2:
                    group = h5f.create_group(f"{key[0]}/{key[1]}")
                elif len(key) == 3:
                    group = h5f.create_group(f"{key[0]}/{key[1]}/{key[2]}")
                elif len(key) == 4:
                    group = h5f.create_group(f"{key[0]}/{key[1]}/{key[2]}/{key[3]}")
                if value is None:
                    continue
                group.create_dataset('data', data=value, compression='gzip')

        print(f'Dictionary saved to {output_path}')

    def load_patches(input_path):
        patches_dict = {}

        def recursive_load(group, keys):
            for key in group.keys():
                new_keys = keys + [key]
                if isinstance(group[key], h5py.Group):
                    recursive_load(group[key], new_keys)
                else:
                    if len(keys) == 2:
                        #Lat, Lon
                        key_tuple = (float(new_keys[0]), float(new_keys[1]))
                    elif len(keys) == 3:
                        # Year, Lat, Lon
                        key_tuple = (int(new_keys[0]), float(new_keys[1]), float(new_keys[2]))
                    elif len(keys) == 4:
                        # Year, Month, Lat, Lon
                        key_tuple = (int(new_keys[0]), int(new_keys[1]), float(new_keys[2]), float(new_keys[3]))
                    else:
                        raise ValueError(f"Unexpected key length: {len(keys)}")
                    
                    patches_dict[key_tuple] = group[key][:]

        with h5py.File(input_path, 'r') as h5f:
            recursive_load(h5f, [])

        return patches_dict

    def save_csv(arr, column_names, output_path):    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(arr, columns=column_names)
        df.to_csv(output_path, index=False)
        print(f'\Csv saved to {output_path}\n')