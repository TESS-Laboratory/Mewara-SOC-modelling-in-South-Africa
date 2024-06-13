import os
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.merge import merge
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

class data_utils:
    @staticmethod
    def lat_lon_to_pixel(dataset, lat, lon):
        transform = dataset.transform
        pixel_x, pixel_y = ~transform * (lon, lat)
        return int(pixel_x), int(pixel_y)

    @staticmethod
    def extract_patch(dataset, lat, lon, patch_size):
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
            raise ValueError("Coordinates are outside the South African boundary")

        pixel_x, pixel_y = data_utils.lat_lon_to_pixel(dataset, lat, lon)
        half_patch = patch_size // 2

        window = Window(pixel_x - half_patch, pixel_y - half_patch, patch_size, patch_size)

        # Read the data within the window
        patch = dataset.read(window=window)

        if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
            pad_width = ((0, 0), 
                         (0, patch_size - patch.shape[1]), 
                         (0, patch_size - patch.shape[2]))
            patch = np.pad(patch, pad_width, mode='constant', constant_values=np.nan)

        return np.squeeze(patch)
    
    def replace_nan_inf_data(data):
        # Calculate mean values ignoring NaN and inf
        data_mean = np.nanmean(data, axis=(0, 1))
        data_mean[np.isnan(data_mean)] = 0  # Handle cases where all values are NaN
        
        # Replace NaN and inf with mean values
        mask_nan = np.isnan(data)
        mask_inf = np.isinf(data)
        for i in range(data.shape[-1]):
            data[mask_nan[..., i], i] = data_mean[i]
            data[mask_inf[..., i], i] = data_mean[i]
    
    def get_landsat_data(year, lat_lon_pairs, patch_size):
        landsat_samples = []
        raster_path=f'Data\LandSat\Annual_Processed\{year}\Landsat_{year}.tif'
        with rasterio.open(raster_path) as dataset:
            for lat, lon in lat_lon_pairs:
                patch = data_utils.extract_patch(dataset, lat, lon, patch_size)
                red_band = patch[0]
                ndvi_band = patch[4]
                rvi_band = patch[7]
                stacked_patch = np.stack([red_band, ndvi_band, rvi_band], axis=-1)
                data_utils.replace_nan_inf_data(stacked_patch)
                landsat_samples.append(stacked_patch)
        return landsat_samples
    
    def get_monthly_climate_data(year, month, lat_lon_pairs, patch_size):
        prec_raster_path =f'Data\WorldClim_SA\wc2.1_2.5m_prec_{year}-{month:02d}.tif'
        tmin_raster_path = f'Data\WorldClim_SA\wc2.1_2.5m_tmin_{year}-{month:02d}.tif'
        tmax_raster_path = f'Data\WorldClim_SA\wc2.1_2.5m_tmax_{year}-{month:02d}.tif'
       
        monthly_climate_samples = []
        
        with rasterio.open(prec_raster_path) as prec_dataset, \
             rasterio.open(tmin_raster_path) as tmin_dataset, \
             rasterio.open(tmax_raster_path) as tmax_dataset: \
             
            for lat, lon in lat_lon_pairs:
                prec_patch = data_utils.extract_patch(prec_dataset, lat, lon, patch_size)
                tmin_patch = data_utils.extract_patch(tmin_dataset, lat, lon, patch_size)
                tmax_patch = data_utils.extract_patch(tmax_dataset, lat, lon, patch_size)
                stacked_patch = np.stack([prec_patch, tmin_patch, tmax_patch], axis=-1)
                data_utils.replace_nan_inf_data(stacked_patch)
                monthly_climate_samples.append(stacked_patch)

        return monthly_climate_samples
    
    def get_terrain_data(lat_lon_pairs, patch_size):
        dem_path =f'Data\TerrainData\Elevation\DEM.tif'
        aspect_path = f'Data\TerrainData\Elevation\Aspect.tif'
        twi_path = f'Data\TerrainData\Elevation\TWI.tif'
        
        terrain_samples = []

        with rasterio.open(dem_path) as dem_dataset, \
             rasterio.open(aspect_path) as aspect_dataset, \
             rasterio.open(twi_path) as twi_dataset:
            
            for lat, lon in lat_lon_pairs:
                dem_patch = data_utils.extract_patch(dem_dataset, lat, lon, patch_size)
                aspect_patch = data_utils.extract_patch(aspect_dataset, lat, lon, patch_size)
                twi_patch = data_utils.extract_patch(twi_dataset, lat, lon, patch_size)
                stacked_patch = np.stack([dem_patch, aspect_patch, twi_patch], axis=-1)
                data_utils.replace_nan_inf_data(stacked_patch)
                terrain_samples.append(stacked_patch)

        return terrain_samples
     
    def get_training_data(soc_data_path, start_year, end_year, patch_size):
        soc_data = pd.read_csv(soc_data_path)
        
        landsat_data = []
        climate_data = []
        terrain_data = []
        targets = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                soc_data_monthly = soc_data[(soc_data['Year'] == year) & (soc_data['Month'] == month)]
                lat_lon_pairs_monthly = list(zip(soc_data_monthly['Lat'], soc_data_monthly['Lon']))
                climate_data_monthly = data_utils.get_monthly_climate_data(year=year, month=month, lat_lon_pairs=lat_lon_pairs_monthly, patch_size=patch_size)
                landsat_data_monthly = data_utils.get_landsat_data(year=year, lat_lon_pairs=lat_lon_pairs_monthly, patch_size=patch_size)
                terrain_data_monthly = data_utils.get_terrain_data(lat_lon_pairs=lat_lon_pairs_monthly, patch_size=patch_size)
               
                for idx in range(len(lat_lon_pairs_monthly)):
                    c_percent = soc_data_monthly.iloc[idx]['C']
                    if idx < len(landsat_data_monthly) and idx < len(climate_data_monthly) and idx < len(terrain_data_monthly):
                        landsat_data.append(landsat_data_monthly[idx])
                        climate_data.append(climate_data_monthly[idx])
                        terrain_data.append(terrain_data_monthly[idx])
                        targets.append(round(c_percent, 2))
        
        return np.array(landsat_data), np.array(climate_data), np.array(terrain_data), np.array(targets)

#data_utils.merge_rasters(r'Data\LandSat\Annual\2007', 'Landsat_2007.tif', r'Data\LandSat\Annual_Processed\2007')