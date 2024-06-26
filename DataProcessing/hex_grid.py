import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt

# Define South Africa bounding box
south_africa_bbox = Polygon([(16.458, -34.834), (32.893, -34.834), (32.893, -22.125), (16.458, -22.125)])

# Create hexagonal grid function
def create_hex_grid(bbox, hex_size):
    minx, miny, maxx, maxy = bbox.bounds
    width = hex_size * 2
    height = np.sqrt(3) / 2 * width
    cols = int(np.ceil((maxx - minx) / width)) + 1
    rows = int(np.ceil((maxy - miny) / height)) + 1
    hexagons = []

    for row in range(rows):
        for col in range(cols):
            x = col * width
            y = row * height
            if col % 2 == 1:
                y += height / 2

            hexagon = Polygon([
                (x, y),
                (x + hex_size, y),
                (x + 1.5 * hex_size, y + height / 2),
                (x + hex_size, y + height),
                (x, y + height),
                (x - 0.5 * hex_size, y + height / 2)
            ])

            if hexagon.intersects(bbox):
                hexagons.append(hexagon)

    hex_grid = gpd.GeoDataFrame(geometry=hexagons, crs='EPSG:4326')
    hex_grid['Hex_Center_Lat'] = hex_grid.centroid.y
    hex_grid['Hex_Center_Lon'] = hex_grid.centroid.x
    
    return hex_grid

# Create hexagonal grid
hex_size = 0.1  # Adjust the hex size as needed
hex_grid = create_hex_grid(south_africa_bbox, hex_size)

# Read soil sample data
# Ensure your soil sample CSV has columns: Year, Month, Lat, Lon, Carbon
soil_samples = gpd.read_file(r'DataProcessing/soc_gdf.csv', GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")

# Convert soil samples to GeoDataFrame
soil_samples['geometry'] = [Point(xy) for xy in zip(soil_samples['Lon'], soil_samples['Lat'])]
soil_samples = gpd.GeoDataFrame(soil_samples, geometry='geometry', crs='EPSG:4326')

# Perform spatial join
joined = gpd.sjoin(soil_samples, hex_grid, how='left', op='within')

# Plot the result
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
hex_grid.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
soil_samples.plot(ax=ax, color='red', markersize=5)
plt.show()

# Save the result
joined.to_file("soc_hex_grid.geojson", driver='GeoJSON')
