import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from rasterio.warp import calculate_default_transform, reproject, Resampling
from DataPreprocessing.raster_preprocessor import data_processor

# Function to calculate slope
def calculate_slope(dem):
    x, y = np.gradient(dem)
    slope = np.sqrt(x**2 + y**2)
    return slope

# Function to calculate aspect
def calculate_aspect(dem):
    x, y = np.gradient(dem)
    aspect = np.arctan2(-x, y)
    return aspect

# Function to calculate TWI
def calculate_twi(dem, slope):
    area = gaussian_filter(dem, sigma=1)
    twi = np.log(area / np.tan(slope))
    return twi

def resample_raster(src_path, dst_path, scale_factor):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width // scale_factor, src.height // scale_factor, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': src.crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )

def clip_dem_to_sa(dem_path, south_africa, output_path):
    # Open the DEM file
    with rasterio.open(dem_path) as src:
        # Clip the DEM to the South Africa boundary
        clipped, transform = mask(src, south_africa.geometry, crop=True)

        # Update metadata
        out_meta = src.meta
        out_meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform
        })

        # Write the clipped DEM to a new GeoTIFF file
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(clipped)

def save_terrain_data():
    output_dir = r"Data/TerrainData"
    os.makedirs(output_dir, exist_ok=True)
    elevation_folder = r'Data/TerrainData/Elevation2'
    south_africa = data_processor.get_sa_shape()

    tile_files = []
    for file in os.listdir(elevation_folder):
        filename = os.fsdecode(file)
        if filename.endswith('.tif'):
            tile_path = os.path.join(elevation_folder, filename)
            #resampled_tile_output = os.path.join(output_dir, f"resampled_{filename}")
            #resample_raster(tile_path, resampled_tile_output, scale_factor=1)
            tile_files.append(tile_path)

    # Merge all resampled tiles
    mosaic_path = os.path.join(output_dir, 'mosaic.tif')
    with rasterio.open(tile_files[0]) as src:
        mosaic, out_trans = merge(tile_files)
        profile = src.profile.copy()

        # Update metadata
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        # Write the merged mosaic to a new GeoTIFF file
        with rasterio.open(mosaic_path, "w", **profile) as dst:
            dst.write(mosaic)

    dem_path = os.path.join(output_dir, 'dem.tif')

    #Clip the merged DEM to South Africa boundary
    #clip_dem_to_sa(mosaic_path, south_africa, clipped_dem_path)

    # Calculate terrain metrics for the clipped DEM
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile

        # Calculate terrain metrics
        slope = calculate_slope(dem_data)
        aspect = calculate_aspect(dem_data)
        twi = calculate_twi(dem_data, slope)

        # Save DEM, slope, aspect, and TWI data to GeoTIFF files
        for data, name in zip([dem_data, slope, aspect, twi], ['DEM', 'Slope', 'Aspect', 'TWI']):
            profile.update(dtype=rasterio.float32, count=1)
            output_path = os.path.join(output_dir, f"{name}.tif")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data.astype(rasterio.float32), 1)

save_terrain_data()