import geopandas as gpd
import rasterio
from rasterio.mask import mask

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