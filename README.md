# Mewara-SOC-Modelling-in-South-Africa
This code was developed by Swati Mewara while undertaking her dissertation project titled 'Mapping Soil Organic Carbon (SOC) Stocks of South Africa using Convolutional Neural Network and Analyzing Trends in SOC Changes'.

The purpose of this repo is to create robust and accurate Soil Organic Carbon (SOC) stock maps for South Africa (SA) from 2000 to 2023 using Convolutional Neural Networks (CNNs) and compare their performance against Random Forest (RF) model. After generating the maps, the trends in SOC stock changes across different biomes in South Africa are generated.

### Data

The data folder contains the following data sources:

- **Field Samples**: Ground soil samples include Date, Latitude, Longitude (in decimals), Carbon (C) percentage by mass, and Bulk Density (BD) in g/cm³. Missing bulk density values are filled using the African iSDA raster dataset [1] during the data processing step. Additional field samples from iSDA are included for modeling, but only a few samples from South Africa are used, while others from Southern African countries are excluded.

### Data Collection

This folder contains Python scripts for collecting covariates

  - `landsat_earth_engine`: Downloads annual Landsat composites from Landsat 5, 7, 8, 9 sensors (1986-2023) from Google Earth Engine [2] to Google Storage Drive. The script harmonizes Landsat bands across different sensors, computes indices like NDVI, SAVI, RVI, and EVI, and saves the annual composites to Google Drive.
  - `raster_preprocessor`: Resamples and merges rasters into a single composite to manage large sizes.
  - `terrain_data`: terrain raster is manually downloaded from Open Topography, which provides access to the Copernicus Digital Elevation Model [3] in chunks for SA. The script downsamples the raster to 120 m resolution and calculates aspect, slope, and total wetness index.
  - `world_clim`: historical weather data until 2021, including monthly average maximum and minimum temperatures in Celsius and monthly total precipitation in millimeters, is manually downloaded from WorldClim [4]. For years 2022 and 2023, total monthly precipitation data was downloaded from CHIRPS [5]. The script clips the raster to the South Africa boundary.

### Data Processing

This folder contains Python scripts for processing field samples for modeling.

- `soc_grid`: Merges field samples from different CSV files into a Python dataframe, removes duplicates, fills missing BD from iSDA, and removes data where carbon is > 60%. Calculates SOC stock and divides South Africa into a hex grid of 100 sq. meters. Each soil sample is assigned to a hex grid to represent the carbon percentage for that area. The final file is saved as `soc_hex_grid.csv`.

### Model

This folder contains the CNN and RF models and utility tools to prepare training data by combining field samples with covariates such as Landsat, climate, and terrain data.

- `training_data_utils`: Extracts raster data from Landsat, climate, and terrain corresponding to field samples for model training. Also includes methods to fetch raster values for test data where lat/lon are the centroids of 100 sq. meter hexagonal grids, totaling 3,500 lat/lon pairs across SA.
- `GridSearchTuner`: Finds the optimal hyperparameters for the RF model.
- `KerasTuner`: Finds the optimal hyperparameter configuration for the CNN model.
- `data_analysis`: Performs Pearson's correlation for each covariate used in modeling to understand its relationship with SOC.
- `base_data_utils`: Distributes the training data spatially into training, validation, and test sets using KMeans clustering. The training set is split in two ways: one by leaving out one cluster for testing and another by spatially distributing validation and test lat/lons near the training lat/lons.
- `base_model`: Common utility code for fetching training data for models from the cache or from the source rasters if not found. It also performs k-fold training for both models.

### MapsTrends

This folder contains scripts for generating SOC stocks map for a given year. It consists of code that fetches covariates from hexagonal grids, with a total of 3,500 centroid points across South Africa. It calls the respective model (RF or CNN) predict method to predict the carbon percentage for each latitude and longitude pair of the centroid of the hexagonal grids.

- `map_utils`: Predicts SOC percentage for a given year and generates corresponding SOC stock map, saving them as PNG file. It also creates scatter plots between predicted and target carbon for the grid points with known target carbon to facilitate analysis of model performance on prediction data.
- `test_metrics`: Fetches the covariates for prediction for the given year, predicts the percentage carbon, calculates the SOC stock based on predicted carbon percent and saves the predictions in CSV format. 
- `plot_utils`: Generates plots for all the predicted SOC stocks.
- `trends_analysis`: Divides the predictions by biomes and calculating the Sen slope for each prediction point over the period from 2000 to 2023. It also calculates the absolute annual SOC stock change (kg C/m²) and the relative SOC stock change (%) compared to the long-term mean SOC. Finally, the 5th, 50th, and 95th percentiles are computed for SOC stock changes, both in relative and absolute terms, to evaluate the trends in SOC stock change within each biome.

### References
1. Miller, M., 2023. African open soil data. OSF, [online] 16 March. Available at: https://doi.org/10.17605/OSF.IO/A69R5 [Accessed 18 June 2024]. 

2. Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D. and Moore, R. (2017) 'Google Earth Engine: Planetary-scale geospatial analysis for everyone', Remote Sensing of Environment. Available at: https://www.worldclim.org/data/monthlywth.html (Accessed: 19 July 2024).

3. European Space Agency and Sinergise (2021) 'Copernicus Global Digital Elevation Model', OpenTopography. Available at: https://doi.org/10.5069/G9028PQB (Accessed: 19 July 2024).

4. WorldClim (no date) 'WorldClim: Monthly Climate Data'. Available at: https://www.worldclim.org/data/monthlywth.html (Accessed: 19 July 2024).

5. Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A. and Michaelsen, J. (2015) 'The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes', Scientific Data, 2(1), pp. 1-21. Available at: https://www.chc.ucsb.edu/data/chirps (Accessed: 19 July 2024).
