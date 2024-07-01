import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Polygon
from rasterio.mask import mask
from shapely.geometry import Point, Polygon
from shapely import wkt

class grid_utils:
    @staticmethod
    def get_sa_bbox():
        sa_bbox = Polygon([(16.458, -34.834), (32.893, -34.834), (32.893, -22.125), (16.458, -22.125)])
        return sa_bbox

    @staticmethod
    def get_sa_shape():
        # Load South Africa shapefile
        shapefile_path = r"Data/SouthAfrica/south_africa_South_Africa_Country_Boundary.shp"
        south_africa_shape = gpd.read_file(shapefile_path)
        return south_africa_shape

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
    def get_carbon_mapping():
        carbon_mapping = {
            "<0.5": "red",
            "0.5-1": "orange",
            "1-2": "yellow",
            "2-3": "green",
            "3-4": "blue",
            ">4": "darkgreen"
        }
        return carbon_mapping

    @staticmethod
    def get_soil_data(soil_csv_path):
        soil_samples = gpd.read_file(soil_csv_path, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")

        soil_samples['C'] = soil_samples['C'].astype(float)
        soil_samples['Lon'] = soil_samples['Lon'].astype(float)
        soil_samples['Lat'] = soil_samples['Lat'].astype(float)

        return soil_samples

    @staticmethod
    def get_hex_grid(hex_size):
        sa_shape = grid_utils.get_sa_shape()

        # Get the bounding box for South Africa in geographic CRS (EPSG:4326)
        bbox = grid_utils.get_sa_bbox()

        minx, miny, maxx, maxy = bbox.bounds

        width = hex_size * 2
        height = np.sqrt(3) * hex_size
        cols = int(np.ceil((maxx - minx) / width)) + 1
        rows = int(np.ceil((maxy - miny) / height)) + 1

        hexagons = []
        hex_ids = []

        for row in range(rows):
            for col in range(cols):
                x = minx + col * width
                y = miny + row * height
                if col % 2 == 1:
                    y += height / 2

                hexagon = Polygon([
                    (x, y),
                    (x + hex_size, y + height / 2),
                    (x + hex_size, y + 1.5 * hex_size),
                    (x, y + 2 * hex_size),
                    (x - hex_size, y + 1.5 * hex_size),
                    (x - hex_size, y + height / 2)
                ])

                if hexagon.intersects(sa_shape.unary_union):
                    hexagons.append(hexagon)
                    hex_ids.append(f"hex_{row}_{col}")

        hex_grid = gpd.GeoDataFrame({'geometry': hexagons, 'Hex_ID': hex_ids}, crs='EPSG:4326')

        # Calculate centroid coordinates for the hex grid
        hex_grid['Hex_Center_Lat'] = hex_grid.centroid.y
        hex_grid['Hex_Center_Lon'] = hex_grid.centroid.x

        # Filter out invalid geometries
        hex_grid = hex_grid[hex_grid.is_valid]

        return hex_grid
    
    @staticmethod
    def get_geoframe(df_with_geometry, geometry_col):
        df_with_geometry[geometry_col] = df_with_geometry[geometry_col].apply(wkt.loads)

        # Filter out invalid geometries
        df_with_geometry = df_with_geometry[df_with_geometry[geometry_col].apply(lambda geom: geom.is_valid)]

        gdf = gpd.GeoDataFrame(df_with_geometry, geometry=geometry_col, crs='EPSG:4326')
        gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    
    @staticmethod
    def get_soc_hex_grid(soil_data, hex_grid_df):
        # Drop rows with missing Latitude and Longitude
        soil_data = soil_data.dropna(subset=['Lat', 'Lon'])

        # Create geometry column from Lat and Lon
        soil_data['geometry'] = [Point(xy) for xy in zip(soil_data['Lon'], soil_data['Lat'])]
        soil_data = gpd.GeoDataFrame(soil_data, geometry='geometry', crs='EPSG:4326')
        hex_grid = grid_utils.get_geoframe(hex_grid_df, 'geometry')

        # Perform spatial join
        joined = gpd.sjoin(soil_data, hex_grid[['Hex_ID', 'geometry']], how='left', predicate='within')
        
        joined.drop(columns='index_right', inplace=True)

        joined.reset_index(drop=True, inplace=True)

        return joined
 
    @staticmethod
    def get_avg_c_each_grid(df, group_by_cols, avg_col, avgc_col_rename):
        carbon_mapping = grid_utils.get_carbon_mapping()
        hex_agg = df.groupby(group_by_cols).agg({avg_col: 'mean'}).reset_index().rename(columns={avg_col: avgc_col_rename})
        
        # Categorize aggregated carbon content
        bins = [-np.inf, 0.5, 1, 2, 3, 4, np.inf]
        labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"]
        hex_agg['C_range'] = pd.cut(hex_agg['Avg_C'], bins=bins, labels=labels)

        # Map colors to 'C_range' values
        hex_agg['Color'] = hex_agg['C_range'].map(carbon_mapping).astype(str)

        # Convert categorical column to string for saving
        hex_agg['C_range'] = hex_agg['C_range'].astype(str)

        return hex_agg

    @staticmethod
    def plot_heat_map(data_avg_c_color_geometry, title, savePlot = False, output_plot_path='', geometry_col='geometry'):
        carbon_mapping = grid_utils.get_carbon_mapping()

        # Plot the result
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        ax.set_aspect('equal')

        # Set plot bounds to ensure aspect ratio is correctly calculated
        minx, miny, maxx, maxy = grid_utils.get_sa_bbox().bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)

        # Plot South Africa boundary with thicker line
        grid_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')

        data_avg_c_color_geometry = data_avg_c_color_geometry.set_geometry(geometry_col)

        # Drop nan
        joined_minus_nan = data_avg_c_color_geometry.dropna(subset=['Avg_C', 'Color'])

        # Plot the grid with the appropriate colors
        joined_minus_nan.plot(ax=ax, color=joined_minus_nan['Color'])

        # Create custom legend
        handles = [Patch(color=color, label=label) for label, color in carbon_mapping.items()]
        legend = ax.legend(handles=handles, title=r'Carbon (% by mass)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=16)

        if savePlot:
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            plt.savefig(output_plot_path, bbox_inches='tight', bbox_extra_artists=[legend])
