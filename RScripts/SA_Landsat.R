#install.packages("rgee")
#install.packages("sf")
#install.packages("dplyr")

library(rgee)
library(sf)
library(dplyr)

ee_Initialize()

south_africa <- ne_countries(country = "South Africa", returnclass = "sf")
south_africa_ee <- sf_as_ee(south_africa)

start_date <- "1986-01-01"
end_date <- "2023-12-31"

#Fetch Landsat Data with Maximum Cloud Cover of 10%:
# Define Landsat collections
landsat_collections <- list(
  ee$ImageCollection("LANDSAT/LT05/C01/T1_SR"), # Landsat 5
  ee$ImageCollection("LANDSAT/LE07/C01/T1_SR"), # Landsat 7
  ee$ImageCollection("LANDSAT/LC08/C01/T1_SR"), # Landsat 8
  ee$ImageCollection("LANDSAT/LC09/C01/T1_SR")  # Landsat 9
)

# Function to filter and select bands
process_landsat_collection <- function(collection) {
  collection$
    filterBounds(south_africa_ee)$
    filterDate(start_date, end_date)$
    filter(ee$Filter$lte("CLOUD_COVER", 10))$
    select(c("B3", "B2", "B1", "B4"))$ # Bands: Blue, Green, Red, NIR
    map(function(image) {
      year_month <- ee$Date(image$get("system:time_start"))$format("YYYY-MM")
      image$set("year_month", year_month)
    })
}

# Apply the function to each collection
landsat_data <- lapply(landsat_collections, process_landsat_collection)

# Merge the collections into one
landsat_combined <- ee$ImageCollection$fromImages(
  lapply(landsat_data, function(x) x$toList(x$size())$getInfo())
)

# Define export parameters
export_params <- list(
  collection = monthly_averages,
  description = "landsat_images",
  folder = "Landsat_South_Africa",
  scale = 30,
  region = south_africa_ee$geometry(),
  maxPixels = 1e13
)

# Export to Google Drive
task <- ee_export_image_collection_to_drive(export_params)
task$start()
ee_monitoring(task)

# Group by year_month and calculate mean
monthly_averages <- landsat_combined$
  filterDate(start_date, end_date)$
  groupBy("year_month")$
  map(function(image) {
    image$reduce(ee$Reducer$mean())
  })

monthly_averages_local <- ee_as_raster(
  image = monthly_averages,
  region = south_africa_ee$geometry(),
  scale = 30
)

# Convert raster to data frame
monthly_averages_df <- as.data.frame(raster::rasterToPoints(monthly_averages_local))

# Save it locally
write.csv(monthly_averages_df, file = "C:\\swati\\Mewara-SOC-modelling-in-South-Africa\\Data\\LandSat\\monthly_averages_landsat.csv", row.names = FALSE)

ggplot(monthly_averages_df, aes(x = x, y = y, fill = B4_mean)) + # Adjust band as needed
  geom_tile() +
  coord_equal() +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(title = "Monthly Averaged NIR Band for South Africa",
       x = "Longitude",
       y = "Latitude",
       fill = "NIR")

