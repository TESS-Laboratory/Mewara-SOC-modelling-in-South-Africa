import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from rasterio.warp import calculate_default_transform, reproject, Resampling

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

def save_terrain_data():
    output_dir = r"Data\TerrainData"
    os.makedirs(output_dir, exist_ok=True)

    dem_path = r'Data\TerrainData\mosaic.tif'
    mosaic_path = r'Data\TerrainData\DEM.tif'
    resample_raster(src_path=dem_path, dst_path=mosaic_path, scale_factor=4)
    
    with rasterio.open(mosaic_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile

        # Calculate terrain metrics
        slope = calculate_slope(dem_data)
        aspect = calculate_aspect(dem_data)
        twi = calculate_twi(dem_data, slope)

        # Save DEM, slope, aspect, and TWI data to GeoTIFF files
        for data, name in zip([slope, aspect, twi], ['Slope', 'Aspect', 'TWI']):
            profile.update(dtype=rasterio.float64, count=1)
            output_path = os.path.join(output_dir, f"{name}.tif")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data.astype(rasterio.float64), 1)
