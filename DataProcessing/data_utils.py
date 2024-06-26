import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon

class data_utils:
    @staticmethod
    def get_sa_bbox():
        sa_bbox = Polygon([(16.458, -34.834), (32.893, -34.834), (32.893, -22.125), (16.458, -22.125)])
        return sa_bbox

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
        # Get the bounding box for South Africa in geographic CRS (EPSG:4326)
        bbox = data_utils.get_sa_bbox()
        bbox = gpd.GeoSeries([bbox], crs='EPSG:4326')
        
        # Re-project the bounding box to a projected CRS (e.g., EPSG:3857)
        #bbox = bbox.to_crs(epsg=3857)
        
        minx, miny, maxx, maxy = bbox.total_bounds

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

                if hexagon.intersects(bbox.unary_union):
                    hexagons.append(hexagon)
                    hex_ids.append(f"hex_{row}_{col}")

        hex_grid = gpd.GeoDataFrame({'geometry': hexagons, 'Hex_ID': hex_ids}, crs='EPSG:4326')

        # Re-project the hex grid back to geographic CRS (EPSG:4326)
        #hex_grid = hex_grid.to_crs(epsg=4326)

        # Calculate centroid coordinates for the hex grid
        hex_grid['Hex_Center_Lat'] = hex_grid.centroid.y
        hex_grid['Hex_Center_Lon'] = hex_grid.centroid.x

        # Filter out invalid geometries
        hex_grid = hex_grid[hex_grid.is_valid]

        return hex_grid
    
    @staticmethod
    def get_hex_grid_avg_carbon_color(hex_grid, carbon_mapping, soil_data):
        # Drop rows with missing Latitude and Longitude
        soil_data = soil_data.dropna(subset=['Lat', 'Lon'])
        
        # Create geometry column from Lat and Lon
        soil_data['geometry'] = [Point(xy) for xy in zip(soil_data['Lon'], soil_data['Lat'])]
        soil_data = gpd.GeoDataFrame(soil_data, geometry='geometry', crs='EPSG:4326')
        
        # Re-project both soil_data and hex_grid to a projected CRS
        soil_data = soil_data.to_crs(epsg=3857)
        hex_grid = hex_grid.to_crs(epsg=3857)

        # Perform spatial join
        joined = gpd.sjoin(soil_data, hex_grid[['Hex_ID', 'geometry']], how='left', predicate='within')

        # Aggregate average carbon content by Hex_ID
        hex_agg = joined.groupby('Hex_ID').agg({'C': 'mean'}).reset_index().rename(columns={'C': 'Avg_C'})
        joined = joined.merge(hex_agg, on='Hex_ID', how='left')

        # Categorize aggregated carbon content
        bins = [-np.inf, 0.5, 1, 2, 3, 4, np.inf]
        labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"]
        joined['C_range'] = pd.cut(joined['Avg_C'], bins=bins, labels=labels)

        # Map colors to 'C_range' values
        joined['color'] = joined['C_range'].map(carbon_mapping).astype(str)

        # Convert categorical column to string for saving
        joined['C_range'] = joined['C_range'].astype(str)

        # Drop rows with missing Avg_C
        joined = joined.dropna(subset=['Avg_C'])

        # Calculate centroid coordinates for hex grid
        joined['Hex_Center_Lat'] = joined.centroid.y
        joined['Hex_Center_Lon'] = joined.centroid.x

        # Re-project hex_grid back to original CRS if needed
        hex_grid = hex_grid.to_crs(epsg=4326)
        joined = joined.to_crs(epsg=4326)

        return joined    
    
    @staticmethod
    def get_square_grid(grid_size_in_meters):
        # Get the bounding box for South Africa in geographic CRS (EPSG:4326)
        bbox = data_utils.get_sa_bbox()
        bbox = gpd.GeoSeries([bbox], crs='EPSG:4326')
        
        # Re-project the bounding box to a projected CRS (e.g., EPSG:3857)
        bbox = bbox.to_crs(epsg=3857)
        minx, miny, maxx, maxy = bbox.total_bounds

        width = height = grid_size_in_meters
        cols = int(np.ceil((maxx - minx) / width)) + 1
        rows = int(np.ceil((maxy - miny) / height)) + 1

        squares = []
        square_ids = []

        for row in range(rows):
            for col in range(cols):
                x = minx + col * width
                y = miny + row * height

                square = Polygon([
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                    (x, y)
                ])

                if square.intersects(bbox.unary_union):
                    squares.append(square)
                    square_ids.append(f"square_{row}_{col}")

        square_grid = gpd.GeoDataFrame({'geometry': squares, 'Square_ID': square_ids}, crs='EPSG:3857')

        # Re-project the square grid back to geographic CRS (EPSG:4326)
        square_grid = square_grid.to_crs(epsg=4326)

        # Calculate centroid coordinates for the square grid
        square_grid['Square_Center_Lat'] = square_grid.centroid.y
        square_grid['Square_Center_Lon'] = square_grid.centroid.x

        # Filter out invalid geometries
        square_grid = square_grid[square_grid.is_valid]

        return square_grid
    
    @staticmethod
    def get_square_grid_avg_carbon_color(square_grid, carbon_mapping, soil_data):
        # Drop rows with missing Latitude and Longitude
        soil_data = soil_data.dropna(subset=['Lat', 'Lon'])
        
        # Create geometry column from Lat and Lon
        soil_data['geometry'] = [Point(xy) for xy in zip(soil_data['Lon'], soil_data['Lat'])]
        soil_data = gpd.GeoDataFrame(soil_data, geometry='geometry', crs='EPSG:4326')
        
        # Re-project both soil_data and square_grid to a projected CRS
        soil_data = soil_data.to_crs(epsg=3857)
        square_grid = square_grid.to_crs(epsg=3857)

        # Perform spatial join
        joined = gpd.sjoin(soil_data, square_grid[['Square_ID', 'geometry']], how='left', predicate='within')

        # Aggregate average carbon content by Square_ID
        square_agg = joined.groupby('Square_ID').agg({'C': 'mean'}).reset_index().rename(columns={'C': 'Avg_C'})
        joined = joined.merge(square_agg, on='Square_ID', how='left')

        # Categorize aggregated carbon content
        bins = [-np.inf, 0.5, 1, 2, 3, 4, np.inf]
        labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"]
        joined['C_range'] = pd.cut(joined['Avg_C'], bins=bins, labels=labels)

        # Map colors to 'C_range' values
        joined['color'] = joined['C_range'].map(carbon_mapping).astype(str)

        # Convert categorical column to string for saving
        joined['C_range'] = joined['C_range'].astype(str)

        # Drop rows with missing Avg_C
        joined = joined.dropna(subset=['Avg_C'])

        # Calculate centroid coordinates for the square grid
        joined['Square_Center_Lat'] = joined.centroid.y
        joined['Square_Center_Lon'] = joined.centroid.x

        # Re-project square_grid back to original CRS if needed
        square_grid = square_grid.to_crs(epsg=4326)
        joined = joined.to_crs(epsg=4326)

        return joined

    @staticmethod
    def plot_soil_data_heat_map(soil_data, title, use_square_grid = True, savePlot=False, output_plot_path=''):
        carbon_mapping = data_utils.get_carbon_mapping()
        if use_square_grid:
            grid = data_utils.get_square_grid()
            joined = data_utils.get_square_grid_avg_carbon_color(square_grid=grid, carbon_mapping=carbon_mapping, soil_data=soil_data)
        else:
            grid = data_utils.get_hex_grid(hex_size=0.1)
            joined = data_utils.get_hex_grid_avg_carbon_color(hex_grid=grid, carbon_mapping=carbon_mapping, soil_data=soil_data)
        
        data_utils.plot_heat_map(soil_data_with_avg_c_color=joined, title=title, savePlot=savePlot, output_plot_path=output_plot_path)

    @staticmethod
    def plot_heat_map(soil_data_with_avg_c_color, title, savePlot = False, output_plot_path=''):
        carbon_mapping = data_utils.get_carbon_mapping()

        # Plot the result
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_aspect('equal')

        # Set plot bounds to ensure aspect ratio is correctly calculated
        minx, miny, maxx, maxy = data_utils.get_sa_bbox().bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Plot South Africa boundary with thicker line
        data_utils.get_sa_shape().boundary.plot(ax=ax, linewidth=1, edgecolor='black')

        # Plot the grid with the appropriate colors
        soil_data_with_avg_c_color.plot(ax=ax, color=soil_data_with_avg_c_color['color'])

        # Create custom legend
        handles = [Patch(color=color, label=label) for label, color in carbon_mapping.items()]
        ax.legend(handles=handles, title='C (% by mass)', loc='center left', bbox_to_anchor=(1, 0.5))

        if savePlot:
            plt.savefig(output_plot_path)
        
        plt.show()
