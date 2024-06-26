import os
import geopandas as gpd
from rasterio.merge import merge
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

class data_processor:
    @staticmethod
    def get_sa_shape():
        # Load South Africa shapefile
        south_africa_shapefile = r"Data/SouthAfrica/south_africa_South_Africa_Country_Boundary.shp"
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
    def harmonizeBands(image, bandMap):
        bandNames = list(bandMap.keys())
        newBandNames = [bandMap[key] for key in bandNames]
        return image.select(bandNames, newBandNames)

    def list_files_recursive(input_dir, output_dir):
        files = []
        for root, dirs, file_list in os.walk(input_dir):
            for file in file_list:
                filename = os.fsdecode(file)
                if filename.endswith('.tif'):
                    tile_path = os.path.join(root, file)
                    resampled_folder = f'{output_dir}/resampled'
                    if not os.path.exists(resampled_folder):
                    os.makedirs(resampled_folder)
                    resampled_tile_output = os.path.join(resampled_folder, f"resampled_{filename}")
                    if not os.path.exists(resampled_tile_output):
                        data_processor.resample_raster(tile_path, resampled_tile_output, scale_factor=4)
                    files.append(resampled_tile_output)
        return files

    def resample_raster(src_path, dst_path, scale_factor):
            with rasterio.open(src_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width // scale_factor, src.height // scale_factor, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': src.crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'count': src.count,  # Ensures the number of bands is the same
                    'dtype': 'float64'  # Ensure consistency in data types
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

                    # Copy band names
                    dst.descriptions = src.descriptions

    def merge_rasters(input_dir, output_filename_with_ext, output_dir):
            tile_files = data_processor.list_files_recursive(input_dir=input_dir, output_dir=output_dir)
            print('\n Resampled all the tile_files\n')
            mosaic_path = os.path.join(output_dir, output_filename_with_ext)
            with rasterio.open(tile_files[0]) as src:
                print('\start merging tile_files\n')
                mosaic, out_trans = merge(tile_files)
                print('\nmerged tile_files\n')
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
                    dst.descriptions = src.descriptions
        
    @staticmethod
    def rename_bands(tif_path, output_path, new_band_names):
        with rasterio.open(tif_path) as src:
            # Read the metadata of the source file
            meta = src.meta

            # Update metadata with new band names
            meta.update({
                'dtype': 'float64'  # Ensuring the data type is consistent
            })

            with rasterio.open(output_path, 'w', **meta) as dst:
                # Copy each band data from source to destination
                for i in range(1, src.count + 1):
                    band_data = src.read(i)
                    dst.write(band_data, i)

                # Set new band names
                dst.descriptions = new_band_names

data_processor.merge_rasters(r'Data/LandSat/Annual_Processed/2007/2007/resampled', 'Landsat_2007.tif', r'Data/LandSat/Annual_Processed/2007')
#new_band_names = ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI']
