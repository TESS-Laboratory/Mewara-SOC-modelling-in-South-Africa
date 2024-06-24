import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from matplotlib.patches import Patch
import DataPreprocessing.data_utils

south_africa_bbox = Polygon([(16.458, -34.834), (32.893, -34.834), (32.893, -22.125), (16.458, -22.125)])

south_africa_boundary = DataPreprocessing.data_utils.data_utils.get_sa_shape()

# Define color mapping based on SOC ranges
color_mapping = {
    "<0.5": "red",
    "0.5-1": "orange",
    "1-2": "yellow",
    "2-3": "green",
    "3-4": "blue",
    ">4": "darkgreen"
}

# Create hexagonal grid function
def create_hex_grid(bbox, hex_size):
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

            if hexagon.intersects(bbox):
                hexagons.append(hexagon)
                hex_ids.append(f"hex_{row}_{col}")

    hex_grid = gpd.GeoDataFrame({'geometry': hexagons, 'Hex_ID': hex_ids}, crs='EPSG:4326')
    hex_grid['Hex_Center_Lat'] = hex_grid.centroid.y
    hex_grid['Hex_Center_Lon'] = hex_grid.centroid.x
    
    return hex_grid

# Create hexagonal grid
hex_size = 0.1  # Adjust the hex size as needed
hex_grid = create_hex_grid(south_africa_bbox, hex_size)

# Check for and remove invalid geometries in hex_grid
hex_grid = hex_grid[hex_grid.is_valid]

# Read soil sample data
# Ensure your soil sample CSV has columns: Year, Month, Lat, Lon, Carbon
soil_samples = gpd.read_file(r'DataProcessing\soc_gdf.csv', GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")

soil_samples['C'] = soil_samples['C'].astype(float)
soil_samples['Lon'] = soil_samples['Lon'].astype(float)
soil_samples['Lat'] = soil_samples['Lat'].astype(float)

# Convert soil samples to GeoDataFrame
soil_samples['geometry'] = [Point(xy) for xy in zip(soil_samples['Lon'], soil_samples['Lat'])]
soil_samples = gpd.GeoDataFrame(soil_samples, geometry='geometry', crs='EPSG:4326')

# Perform spatial join
joined = gpd.sjoin(soil_samples, hex_grid[['Hex_ID', 'geometry']], how='left', op='within')

# Average the carbon content C by hex ID for plotting
hex_agg = joined.groupby('Hex_ID').agg({'C': 'mean'}).reset_index()
hex_grid = hex_grid.merge(hex_agg, on='Hex_ID', how='left')
hex_grid = hex_grid.dropna(subset=['C'])

# Categorize aggregated carbon content
bins = [-np.inf, 0.5, 1, 2, 3, 4, np.inf]
labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4"]
hex_grid['C_range'] = pd.cut(hex_grid['C'], bins=bins, labels=labels)

# Map colors to 'C_range' values
hex_grid['color'] = hex_grid['C_range'].map(color_mapping).astype(str)

# Convert categorical column to string for saving
hex_grid['C_range'] = hex_grid['C_range'].astype(str)

joined = joined.merge(hex_grid, on='Hex_ID', how='left')

joined.to_csv(r"DataProcessing\soc_hex_grid.csv", index=False, mode='w')

# Plot the result
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.set_aspect('equal')

# Set plot bounds to ensure aspect ratio is correctly calculated
minx, miny, maxx, maxy = south_africa_bbox.bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_title(f'Long Term Average Carbon (% by Mass) Heat Map for South Africa (1986-2022)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Plot South Africa boundary with thicker line
south_africa_boundary.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

# Plot the hex grid with the appropriate colors
hex_grid.plot(ax=ax, color=hex_grid['color'])

# Create custom legend
handles = [Patch(color=color, label=label) for label, color in color_mapping.items()]
legend = ax.legend(handles=handles, title='C (% by mass)', loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
