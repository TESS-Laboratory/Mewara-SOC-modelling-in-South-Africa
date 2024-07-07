import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from Model.training_data_utils import training_data_utils
import geopandas as gpd
from shapely.geometry import Point
from spatialkfold.clusters import spatial_kfold_clusters 
from spatialkfold.blocks import spatial_blocks

class base_data_utils:
    def get_train_val_test_data(lat_lon_data, landsat_data, climate_data, terrain_data, targets):
        # Split data into training and test sets
        landsat_train, landsat_test, climate_train, climate_test, dem_train, dem_test, targets_train, targets_test = train_test_split(
            landsat_data, climate_data, terrain_data, targets, test_size=0.1, random_state=42)

        # Split data into train and validation sets
        landsat_train, landsat_val, climate_train, climate_val, dem_train, dem_val, targets_train, targets_val = train_test_split(
            landsat_train, climate_train, dem_train, targets_train, test_size=0.1, random_state=42)
        
        return landsat_train, landsat_val, landsat_test, climate_train, climate_val, climate_test, dem_train, dem_val, dem_test, \
        targets_train, targets_val, targets_test

    def spatial_kfold_split(lat_lon_data, landsat_data, climate_data, terrain_data, targets, k_fold):
        n_samples = np.sqrt(landsat_data.shape[0])
        # Create GeoDataFrame from lat_lon_data
        gdf = gpd.GeoDataFrame(lat_lon_data, columns=['lat', 'lon'])
        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        
        # Perform KMeans clustering on the spatial data
        coords = gdf[['lat', 'lon']].values
        kmeans = KMeans(n_clusters=min(50, int(n_samples)), random_state=42)
        gdf['cluster'] = kmeans.fit_predict(coords)
        
        # Create an array of the cluster labels
        clusters = gdf['cluster'].values

        # Function to get the train, validation, and test sets
        def get_split_data(indices):
            return (
                lat_lon_data[indices],
                landsat_data[indices],
                climate_data[indices],
                terrain_data[indices],
                targets[indices]
            )

        train_splits = [[] for _ in range(k_fold)]
        val_splits = [[] for _ in range(k_fold)]
        test_splits = [[] for _ in range(k_fold)]

        # Iterate through each cluster and perform K-Fold split
        for cluster_label in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_label)[0]
            cluster_coords = coords[cluster_indices]

            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            if len(cluster_coords) > 1: 
                cluster_folds = list(kf.split(cluster_coords))
            else:
                cluster_folds = list(cluster_coords)

            for fold_idx in range(k_fold):
                train_idx, val_idx = cluster_folds[fold_idx]
                absolute_train_idx = cluster_indices[train_idx]
                absolute_val_idx = cluster_indices[val_idx]

                # Further split absolute_val_idx into val and test indices
                kv = KFold(n_splits=2, shuffle=True, random_state=42)
                if len(absolute_val_idx) > 1:
                    val_test_idx, test_idx = next(kv.split(absolute_val_idx))
                    absolute_val_test_idx = absolute_val_idx[val_test_idx]
                    absolute_test_idx = absolute_val_idx[test_idx]
                    val_splits[fold_idx].append(get_split_data(absolute_val_test_idx))
                    test_splits[fold_idx].append(get_split_data(absolute_test_idx))
                else:
                    val_splits[fold_idx].append(get_split_data(absolute_val_idx))

                train_splits[fold_idx].append(get_split_data(absolute_train_idx))
                
        # Function to combine splits into single arrays
        def combine_splits(splits):
            combined = [np.concatenate([split[i] for split in splits], axis=0) for i in range(len(splits[0]))]
            return combined

        # Combine the splits for each fold
        folds = []
        for fold_idx in range(k_fold):
            train_data = combine_splits(train_splits[fold_idx])
            val_data = combine_splits(val_splits[fold_idx])
            test_data = combine_splits(test_splits[fold_idx])
            folds.append((train_data, val_data, test_data))

        return folds

    def get_test_train_data(soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, use_cache=True):
        lat_lon_array = []
        landsat_array = []
        climate_array = []
        terrain_array = []
        targets_array = []

        for year in years:
            cache_path = f'Data\Train\Cache\Train_{year}.pkl'

            # Check if cache exists
            if os.path.exists(cache_path):
                print(f"Loading data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    lat_lon_array.extend(data['lat_lon_data'])
                    landsat_array.extend(data['landsat_data'])
                    climate_array.extend(data['climate_data'])
                    terrain_array.extend(data['terrain_data'])
                    targets_array.extend(data['targets'])
            
            else:
                # Fetch the data
                print(f"Fetching data from source for year {year}")
                lat_lon_data, landsat_data, climate_data, terrain_data, targets = training_data_utils.get_training_data(
                    soc_data_path=soc_data_path,
                    years=[year],
                    start_month=start_month,
                    end_month=end_month,
                    patch_size_meters_landsat=patch_size_meters_landsat,
                    patch_size_meters_climate=patch_size_meters_climate,
                    patch_size_meters_terrain=patch_size_meters_terrain
                )

                if use_cache:
                    # Save the data to cache
                    print(f"Saving data to cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'lat_lon_data': lat_lon_data,
                            'landsat_data': landsat_data,
                            'climate_data': climate_data,
                            'terrain_data': terrain_data,
                            'targets': targets
                        }, f)

                lat_lon_array.extend(lat_lon_data)
                landsat_array.extend(landsat_data)
                climate_array.extend(climate_data)
                terrain_array.extend(terrain_data)
                targets_array.extend(targets)

        return lat_lon_array, landsat_array, climate_array, terrain_array, targets_array

    def plot_trainin_validation_loss(train_loss, val_loss):
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()