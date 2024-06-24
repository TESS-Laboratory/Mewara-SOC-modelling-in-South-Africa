import os
import geopandas as gpd
from rasterio.merge import merge
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

class raster_preprocessor:
    @staticmethod
    def harmonizeBands(image, bandMap):
        bandNames = list(bandMap.keys())
        newBandNames = [bandMap[key] for key in bandNames]
        return image.select(bandNames, newBandNames)

    @staticmethod
    def list_files_recursive(input_dir, output_dir):
        files = []
        for root, dirs, file_list in os.walk(input_dir):
            for file in file_list:
                filename = os.fsdecode(file)
                if filename.endswith('.tif'):
                    tile_path = os.path.join(root, file)
                    #resampled_tile_output = os.path.join(f'{output_dir}/resampled', f"resampled_{filename}")
                    #if not os.path.exists(resampled_tile_output):
                        #data_processor.resample_raster(tile_path, resampled_tile_output, scale_factor=4)
                    files.append(tile_path)
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

    @staticmethod
    def merge_rasters(input_dir, output_filename_with_ext, output_dir):
            tile_files = raster_preprocessor.list_files_recursive(input_dir=input_dir, output_dir=output_dir)
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

#data_processor.merge_rasters(r'Data\LandSat\Annual_Processed\2009\resampled', 'Landsat_2009.tif', r'Data\LandSat\Annual_Processed\2009')
new_band_names = ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI']
raster_preprocessor.rename_bands(r'Data\LandSat\Annual_Processed\2017\Landsat_2017.tif', r'Data\LandSat\Annual_Processed\2017\Landsat_2017_new.tif', new_band_names=new_band_names)
#data_processor.resample_raster(src_path=r'C:\Users\Swati Mewara\Downloads\Annual_Composite4_2010-0000023552-0000011776.tif', dst_path=r'C:\Mewara-SOC-modelling-in-South-Africa\Data\LandSat\Annual_Processed\2010\resampled.tif', scale_factor=4)