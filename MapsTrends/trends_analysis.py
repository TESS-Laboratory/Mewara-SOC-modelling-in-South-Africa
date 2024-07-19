import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.linear_model import TheilSenRegressor
import geopandas as gpd

class trends_analysis:
    def get_biome_geometry(biome):
        biome_shapefile_path = f'Data/SouthAfricaBiomes/{biome}_Biome.shp'
        biome_shape = gpd.read_file(biome_shapefile_path)
        return biome_shape.geometry
    
    def get_predictions_by_biome(biome, output_folder, predictions_path, save_csv=True):
        predictions = pd.read_csv(predictions_path)

        predictions = predictions.dropna(subset=['Lat', 'Lon'])
        predictions = predictions[(predictions['Year'] >= 2000) & (predictions['Year'] <= 2023)]

        predictions['geometry'] = [Point(xy) for xy in zip(predictions['Lon'], predictions['Lat'])]
        predictions = gpd.GeoDataFrame(predictions, geometry='geometry', crs='EPSG:4326')
        biome_geometry = trends_analysis.get_biome_geometry(biome)

        biome_gdf = gpd.GeoDataFrame(geometry=biome_geometry, crs='EPSG:4326')

        biome_predictions = gpd.sjoin(predictions, biome_gdf, how='inner')
        
        biome_predictions.drop(columns='index_right', inplace=True)

        biome_predictions.reset_index(drop=True, inplace=True)
        biome_predictions['SOC'] = biome_predictions['C'] * biome_predictions['Target_BD'] * 20

        if save_csv:
            os.makedirs(output_folder, exist_ok=True)
            biome_predictions.to_csv(f'{output_folder}/{biome}_predictions.csv', index=False)

        return biome_predictions
    
    def biome_trends(biome, output_folder, predictions_path):
        biome_predictions = trends_analysis.get_predictions_by_biome(biome=biome, output_folder=output_folder, predictions_path=predictions_path)
        
        pixel_long_term_mean_c = biome_predictions.groupby(['Lat', 'Lon']).agg({'SOC': 'mean'}).reset_index().rename(columns={'SOC': 'Mean_SOC'})
        
        pixel_pred_year = biome_predictions.groupby(['Year', 'Lat', 'Lon']).agg({'SOC': 'mean'}).reset_index().rename(columns={'SOC': 'Mean_SOC_Year'})

        lat_lon_pairs = set(zip(pixel_long_term_mean_c['Lat'], pixel_long_term_mean_c['Lon']))

        soc_changes = []
        
        for lat, lon in lat_lon_pairs:
            long_mean_soc = np.mean(pixel_long_term_mean_c[(pixel_long_term_mean_c['Lat'] == lat) & (pixel_long_term_mean_c['Lon'] == lon)]['Mean_SOC'].values)
            
            pixel_preds = pixel_pred_year[(pixel_pred_year['Lat'] == lat) & (pixel_pred_year['Lon'] == lon)]
            
            coefficient, intercept = trends_analysis.theil_sen_regression(pixel_preds['Year'], pixel_preds['Mean_SOC_Year'])
          
            soc_change_percent = (coefficient / long_mean_soc) * 100
            
            soc_changes.append([lat, lon, long_mean_soc, soc_change_percent])

        soc_df = pd.DataFrame(data=soc_changes, columns=['Lat', 'Lon', 'Mean_SOC', 'SOC_Change_Percent'])
        os.makedirs(output_folder, exist_ok=True)
        soc_df.to_csv(f'{output_folder}/{biome}.csv', index=False)

        net_soc = soc_df['Mean_SOC'].sum()
        net_soc_change_percent = soc_df['SOC_Change_Percent'].mean()
        return biome, net_soc, net_soc_change_percent
    
    def theil_sen_regression(X, y):
        theil_sen = TheilSenRegressor()
        theil_sen.fit(X.values.reshape(-1, 1), y)
        
        coefficient = theil_sen.coef_[0]
        intercept = theil_sen.intercept_
        
        return coefficient, intercept

    def save_biome_trends(predictions_folder, output_folder):
        biomes = ['Forest', 'Fynbos', 'SucculentKaroo', 'NamaKaroo', 'Thicket', 'Savanna', 'Grassland'] 
        biome_trend = []

        for biome in biomes:   
           biome, net_SOC, net_SOC_Change_Percent = trends_analysis.biome_trends(biome, output_folder, f'{predictions_folder}/combined_predictions.csv')
           biome_trend.append([biome, net_SOC, net_SOC_Change_Percent])

        biome_trend_df = pd.DataFrame(biome_trend, columns=['Biome', 'Net_SOC', 'Net_SOC_Change_Percent'])
        os.makedirs(output_folder, exist_ok=True)
        biome_trend_df.to_csv(f'{output_folder}/Biome_Trends.csv', index=False)

#trends_analysis.save_biome_trends('MapsTrends\RF_Model\Predictions', 'MapsTrends\RF_Model\Trends')