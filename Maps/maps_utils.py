import pandas as pd
from Testing.TestMetrics import TestMetrics

class map_utils:
    @staticmethod
    def create_map(year, soc_hex_grid_path, model_path, patch_size_meters_landsat, patch_size_meters_climate, patch_size_meters_terrain):
        soc_hex_grid = pd.read_csv(soc_hex_grid_path)
        lat_lon_pairs = [(soc_hex_grid['hex_center_lat'], soc_hex_grid['hex_center_lon'])]

        test_metrics = TestMetrics(model_path=model_path)

        test_metrics.predict(year=year, soc_grid_path=soc_hex_grid_path, lat_lon_pairs=lat_lon_pairs, patch_size_meters_landsat=patch_size_meters_landsat,
                             patch_size_meters_climate=patch_size_meters_climate, patch_size_meters_terrain=patch_size_meters_terrain,
                             save=True)
        for idx in range(len(lat_lon_pairs)):
            each_lat, each_lon = lat_lon_pairs[idx]