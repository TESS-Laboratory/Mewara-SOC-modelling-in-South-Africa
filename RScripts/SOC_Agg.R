# Load required packages
library(ggplot2)
library(dplyr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)

# Load Africa boundaries
africa <- ne_countries(continent = "Africa", returnclass = "sf")

# Filter for South Africa
south_africa <- africa %>% filter(admin == "South Africa")
south_africa_sf <- st_as_sf(south_africa, crs= st_crs(4326))

# Define color mapping based on SOC ranges
color_mapping <- c("<0.5" = "red", "0.5-1" = "orange", "1-2" = "yellow", 
                   "2-3" = "green", "3-4" = "blue", ">4" = "darkgreen")

# Convert SOC data to an sf object
soc_sf <- st_as_sf(soc_data, coords = c("Lon", "Lat"), crs = 4326)

# Create a hexagonal grid over South Africa
hex_size <- 0.25 # Size of the hexagonal grid cells (degrees)
hex_grid_sf <- st_make_grid(south_africa_sf, cellsize = hex_size, square = FALSE) %>%
  st_sf() %>%
  mutate(ID = row_number())

# Ensure hex_grid_sf has the same CRS
hex_grid_sf <- st_transform(hex_grid_sf, st_crs(south_africa_sf))

# Perform spatial join to intersect points with hexagonal grid cells
soc_hex <- st_join(soc_sf, hex_grid_sf, join = st_intersects)

# Aggregate the 'C' and 'BD' values by hexagonal cells
hex_avg <- soc_hex %>%
  group_by(ID) %>%
  summarise(mean_C = mean(C, na.rm = TRUE), mean_BD = mean(BD, na.rm = TRUE)) %>%
  mutate(mean_C_range = cut(mean_C, breaks = c(-Inf, 0.5, 1, 2, 3, 4, Inf), 
                            labels = c("<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4")),
         mean_BD_range = cut(mean_BD, breaks = c(-Inf, 0.5, 1, 2, 3, 4, Inf), 
                             labels = c("<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4")))

# Function to plot the map
plotMap <- function(data, title, color_Col) {
  ggplot() +
    geom_sf(data = south_africa, fill = NA, color = "black") +  
    geom_sf(data = hex_grid_sf, fill = NA, color = "gray", size = 0.2) + 
    geom_sf(data = data, aes(color = !!sym(color_Col))) +
    scale_color_manual(values = color_mapping) +
    coord_sf() +
    theme_minimal() +
    labs(title = title,
         x = "Longitude",
         y = "Latitude")
}

# Plot the maps
plotMap(hex_avg, "Carbon % by Mass Distribution in South Africa for (1987-2018)", "mean_C_range")
plotMap(hex_avg, "Bulk Density in g/cm3 Distribution in South Africa for (1987-2018)", "mean_BD_range")
