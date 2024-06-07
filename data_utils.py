import os
import geopandas as gpd
from rasterio.merge import merge
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

class data_utils:
    @staticmethod
    def get_sa_shape():
        # Load South Africa shapefile
        south_africa_shapefile = r"Data\SouthAfrica\south_africa_South_Africa_Country_Boundary.shp"
        return gpd.read_file(south_africa_shapefile)
        
    @staticmethod
    def clip_to_sa(rasterfile_path, south_africa, output_path):
        # Open the DEM file
        with rasterio.open(rasterfile_path) as src:
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

    @staticmethod
    def list_files_recursive(input_dir, output_dir):
        files = []
        for root, dirs, file_list in os.walk(input_dir):
            for file in file_list:
                filename = os.fsdecode(file)
                if filename.endswith('.tif'):
                    tile_path = os.path.join(root, file)
                    resampled_tile_output = os.path.join(f'{output_dir}\\resampled', f"resampled_{filename}")
                    data_utils.resample_raster(tile_path, resampled_tile_output, scale_factor=4)
                    files.append(resampled_tile_output)
        return files
    
    @staticmethod
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
    
    @staticmethod
    def merge_rasters(input_dir, output_filename_with_ext, output_dir):
        tile_files = data_utils.list_files_recursive(input_dir=input_dir, output_dir=output_dir)
        mosaic_path = os.path.join(output_dir, output_filename_with_ext)
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

data_utils.merge_rasters(r'Data\Landsat_South_Africa\Annual\1987', 'Landsat_1987.tif', r'Data\Landsat_South_Africa\Annual_Processed\1987')
