library(elevatr)
library(viridis)
library(tidyverse)
library(raster)
library(terra)

# Get elevation data for South Africa
elevation <- elevatr::get_elev_raster(locations = south_africa_sf, z = 10)

# Compute slope and aspect
slope <- terrain(elevation, opt='slope', unit='degrees')
aspect <- terrain(elevation, opt='aspect',unit='degrees')

# Calculate Topographic Wetness Index (TWI)
slope_radians <- terrain(elevation, opt = "slope", unit = "radians")
area <- terrain(elevation, opt = "TPI")
area <- exp(area)
twi <- log(area / tan(slope_radians))

# Stack the rasters
stacked_rasters <- stack(elevation, slope, aspect, twi)
names(stacked_rasters) <- c("Elevation", "Slope", "Aspect", "TWI")
output_folder <- "C:\\swati\\Mewara-SOC-modelling-in-South-Africa\\Data"

# Define file pattern
filename_pattern <- "C:\\swati\\Mewara-SOC-modelling-in-South-Africa\\TerrainData_%s.tif"

# Write the rasters to files
for(i in 1:nlayers(stacked_rasters)) {
  layer <- stacked_rasters[[i]]
  output_filename <- sprintf(filename_pattern, names(stacked_rasters)[i])
  
  # Write the layer to a file
  writeRaster(layer, filename = output_filename, format = "GTiff")
}

# Plotting
par(mfrow = c(2, 2))
plot(elevation, main="DEM (meters) for SA")
plot(slope, main="Slope for SA (degrees)", col=topo.colors(6,alpha=0.6))
plot(aspect, main="Aspect for SA (degrees)", col=topo.colors(6,alpha=0.6))
plot(twi, main="Topographic Wetness Index for SA", col=topo.colors(6, alpha=0.6))