import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from Model.training_data_utils import training_data_utils
import geopandas as gpd
from shapely.geometry import Point
from spatialkfold.clusters import spatial_kfold_clusters 
from spatialkfold.blocks import spatial_blocks

class base_data_utils:
    def spatial_leave_cluster_out_split(lat_lon_data, landsat_data, climate_data, terrain_data, targets):
        # Create GeoDataFrame from lat_lon_data
        gdf = gpd.GeoDataFrame(lat_lon_data, columns=['lat', 'lon'])
        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        
        # Perform KMeans clustering on the spatial data
        coords = gdf[['lat', 'lon']].values
        kmeans = KMeans(n_clusters=50, random_state=42)
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

        train_indices = []
        val_indices = []
        test_indices = []

        # Iterate through each cluster and perform K-Fold split
        # Get unique clusters
        unique_clusters = np.unique(clusters)

        # Randomly select clusters for validation and testing
        val_cluster_label = np.random.choice(unique_clusters, 1)[0]

        for cluster_label in unique_clusters:
            cluster_indices = np.where(clusters == cluster_label)[0]
            if (cluster_label == val_cluster_label):
                if len(cluster_indices) > 1:
                    val_idx, test_idx = train_test_split(cluster_indices, test_size=0.5, shuffle=True, random_state=42)
                    val_indices.extend(val_idx)
                    test_indices.extend(test_idx)
                else:
                    val_indices.extend(cluster_indices)
            else:
                train_indices.extend(cluster_indices)

        train_sets = get_split_data(train_indices)
        val_sets = get_split_data(val_indices)
        test_sets = get_split_data(test_indices)

        return train_sets, val_sets, test_sets

    def spatial_split(lat_lon_data, landsat_data, climate_data, terrain_data, targets):
        # Create GeoDataFrame from lat_lon_data
        gdf = gpd.GeoDataFrame(lat_lon_data, columns=['lat', 'lon'])
        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        
        # Perform KMeans clustering on the spatial data
        coords = gdf[['lat', 'lon']].values
        kmeans = KMeans(n_clusters=50, random_state=42)
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

        train_indices = []
        val_indices = []
        test_indices = []

        # Iterate through each cluster and perform K-Fold split
        for cluster_label in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_label)[0]

            if len(cluster_indices) > 1:
                # Split 99% for training and 2% for validation and test
                train_idx, val_test_idx = train_test_split(cluster_indices, test_size=0.05, shuffle=True, random_state=100)

                # Further split 20% into 10% validation and 10% test
                if len(val_test_idx) > 1:
                    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, shuffle=True, random_state=100)
                else:
                    val_idx = val_test_idx
                    test_idx = val_test_idx

                train_indices.extend(train_idx)
                val_indices.extend(val_idx)
                test_indices.extend(test_idx)
            else:
                train_indices.extend(cluster_indices)
        
        train_sets = get_split_data(train_indices)
        val_sets = get_split_data(val_indices)
        test_sets = get_split_data(test_indices)

        return train_sets, val_sets, test_sets

    def get_test_train_data(soc_data_path, years, start_month, end_month, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain, use_cache, update_cache):
        lat_lon_array = []
        landsat_array = []
        climate_array = []
        terrain_array = []
        targets_array = []

        for year in years:
            cache_path = f'Data/Train/Cache/L{patch_size_meters_landsat}_C{patch_size_meters_climate}_T{patch_size_meters_terrain}/Train_{year}.pkl'
            # Check if cache exists
            if os.path.exists(cache_path) and use_cache:
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

                if update_cache:
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

        return np.array(lat_lon_array), np.array(landsat_array), np.array(climate_array), np.array(terrain_array), np.array(targets_array)

    def plot_trainin_validation(train, val, metric_label):
        epochs = range(1, len(train) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train, 'b', label=f'Training {metric_label}')
        plt.plot(epochs, val, 'r', label=f'Validation {metric_label}')
        plt.title(f'Training and Validation {metric_label}')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metric_label}')
        plt.legend()
        plt.show()
