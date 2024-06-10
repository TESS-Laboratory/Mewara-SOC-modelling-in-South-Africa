import numpy as np
import rasterio
from rasterio.windows import Window

"""
    Convert latitude and longitude to pixel coordinates in the raster.
"""
def lat_lon_to_pixel(dataset, lat, lon):

    transform = dataset.transform
    pixel_x, pixel_y = ~transform * (lon, lat)
    return int(pixel_x), int(pixel_y)

"""
    Extract a patch centered at the given latitude and longitude from the raster.
"""
def extract_patch(dataset, lat, lon, patch_size):
    
    pixel_x, pixel_y = lat_lon_to_pixel(dataset, lat, lon)
    
    # Define the image size to extract
    half_patch = patch_size // 2
    window = Window(pixel_x - half_patch, pixel_y - half_patch, patch_size, patch_size)
    
    # Read the data within the window
    patch = dataset.read(window=window)
    
    return patch

# Open the raster file
def extract_patch(raster_path, lat_lon_pairs, patch_size):
    patches = []
    with rasterio.open(raster_path) as dataset:
        for lat, lon in lat_lon_pairs:
            patch = extract_patch(dataset, lat, lon, patch_size)
            patches.append(patch)
            print(f"Extracted patch at ({lat}, {lon}):")
            print(patch)
    return patches
        
