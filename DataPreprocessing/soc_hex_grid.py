import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contextily as ctx

from data_utils import data_utils

south_africa = data_utils.get_sa_shape()

def preprocess_data():
    soc_data = pd.read_excel(r'Data\FieldSamples\SOC Data from Heidi 20230124 - cleaned_additional.xlsx')

    # Convert SOC data to GeoDataFrame
    geometry = [Point(xy) for xy in zip(soc_data.Lon, soc_data.Lat)]
    soc_gdf = gpd.GeoDataFrame(soc_data, crs="EPSG:4326", geometry=geometry)

    # Create a hexagonal grid over South Africa
    hex_size = 0.5  # Size of the hexagonal grid cells (degrees)
    bounds = south_africa.total_bounds
    xmin, ymin, xmax, ymax = bounds
    hex_height = hex_size * np.sqrt(3)
    hex_width = hex_size * 2

    # Calculate the number of columns and rows
    n_cols = int(np.ceil((xmax - xmin) / hex_width))
    n_rows = int(np.ceil((ymax - ymin) / hex_height))

    # Create hexagon coordinates
    hexagons = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = xmin + col * hex_width
            y = ymin + row * hex_height
            if row % 2 == 0:
                x += hex_width / 2
            hexagon = Polygon([
                (x, y),
                (x + hex_width / 2, y + hex_height / 2),
                (x + hex_width / 2, y + hex_height / 2 + hex_height),
                (x, y + 2 * hex_height),
                (x - hex_width / 2, y + hex_height / 2 + hex_height),
                (x - hex_width / 2, y + hex_height / 2)
            ])
            if hexagon.intersects(south_africa.unary_union):
                hexagons.append(hexagon)

    hex_grid = gpd.GeoDataFrame({'geometry': hexagons})
    hex_grid.crs = south_africa.crs

    # Assign an ID column to the hex cells
    hex_grid['ID'] = range(1, len(hex_grid) + 1)
    hex_grid.to_csv(r"DataPreprocessing\hex_grid.csv", index=False)

    # Perform spatial join to intersect points with hexagonal grid cells
    soc_hex = gpd.sjoin(soc_gdf, hex_grid, how="inner", op='intersects')
    soc_hex.to_csv(r"DataPreprocessing\soc_hex_grid.csv", index=False)

    # Aggregate the 'C' and 'BD' values by hexagonal cells
    hex_avg = soc_hex.groupby('ID').agg({
        'C': 'mean', 'BD': 'mean'
    }).reset_index()

    hex_avg['mean_C_range'] = pd.cut(hex_avg['C'], bins=[-float('inf'), 0.5, 1, 2, 3, 4, float('inf')], labels=["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"])
    hex_avg['mean_BD_range'] = pd.cut(hex_avg['BD'], bins=[-float('inf'), 0.5, 1, 2, 3, 4, float('inf')], labels=["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"])

    # Merge with hex_grid to have geometry in the final dataframe
    hex_avg = hex_grid.merge(hex_avg, on='ID')
    hex_avg.to_csv("soc_avg_hex.csv", index=False)

def plot_map(data, title, color_col):
    color_mapping = {
    "<0.5": "red", "0.5-1": "orange", "1-2": "yellow",
    "2-3": "green", "3-4": "blue", ">4": "darkgreen"
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    south_africa.boundary.plot(ax=ax, linewidth=1)
    data.plot(column=color_col, ax=ax, legend=True, cmap=ListedColormap(color_mapping.values()))
    ctx.add_basemap(ax, crs=south_africa.crs.to_string())
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_mean_avg():
    hex_avg = pd.read_csv('soc_avg_hex.csv')
    plot_map(hex_avg, "Carbon % by Mass Distribution in South Africa for (1987-2018)", "mean_C_range")
    plot_map(hex_avg, "Bulk Density in g/cm3 Distribution in South Africa for (1987-2018)", "mean_BD_range")

preprocess_data()
