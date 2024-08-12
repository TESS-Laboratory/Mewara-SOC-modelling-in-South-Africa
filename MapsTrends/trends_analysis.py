import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.linear_model import TheilSenRegressor
import geopandas as gpd
from MapsTrends.plot_utils import plot_utils

class trends_analysis:
    def get_biome_geometry(biome):
        biome_shapefile_path = f'Data/SouthAfricaBiomes/{biome}_Biome.shp'
        biome_shape = gpd.read_file(biome_shapefile_path)
        return biome_shape.geometry
    
    def get_predictions_by_biome(biome, output_folder, predictions_path, save_csv=False):
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

        if save_csv:
            os.makedirs(output_folder, exist_ok=True)
            biome_predictions.to_csv(f'{output_folder}/{biome}_predictions.csv', index=False)

        return biome_predictions
    
    def biome_trends(biome, output_folder, predictions_path):
        biome_predictions = trends_analysis.get_predictions_by_biome(biome=biome, output_folder=output_folder, predictions_path=predictions_path)
        
        pixel_long_term_mean_soc = biome_predictions.groupby(['Lat', 'Lon']).agg({'SOC': 'mean'}).reset_index().rename(columns={'SOC': 'Mean_SOC'})
        
        pixel_pred_year = biome_predictions.groupby(['Year', 'Lat', 'Lon']).agg({'SOC': 'mean'}).reset_index().rename(columns={'SOC': 'Mean_SOC_Year'})

        lat_lon_pairs = set(zip(pixel_long_term_mean_soc['Lat'], pixel_long_term_mean_soc['Lon']))

        soc_changes = []
        
        for lat, lon in lat_lon_pairs:
            long_mean_soc = np.mean(pixel_long_term_mean_soc[(pixel_long_term_mean_soc['Lat'] == lat) & (pixel_long_term_mean_soc['Lon'] == lon)]['Mean_SOC'].values) * 10 # 1 g/cm2 = 10 kg/m2
            
            pixel_preds = pixel_pred_year[(pixel_pred_year['Lat'] == lat) & (pixel_pred_year['Lon'] == lon)]
            
            coefficient, intercept = trends_analysis.theil_sen_regression(pixel_preds['Year'], pixel_preds['Mean_SOC_Year'] * 10)
          
            soc_change_percent = (coefficient / long_mean_soc) * 100
            
            soc_changes.append([biome, lat, lon, coefficient, long_mean_soc, soc_change_percent])
   
        soc_df = pd.DataFrame(data=soc_changes, columns=['Biome', 'Lat', 'Lon', 'Annual_SOC_Change', 'Mean_SOC', 'SOC_Change_Percent'])

        os.makedirs(output_folder, exist_ok=True)
        #soc_df.to_csv(f'{output_folder}/{biome}_Trend.csv', index=False)

        return biome, soc_df
    
    def theil_sen_regression(X, y):
        x = X.values.reshape(-1, 1)
        theil_sen = TheilSenRegressor()
        theil_sen.fit(x, y)
        
        coefficient = theil_sen.coef_[0]
        intercept = theil_sen.intercept_
        
        return coefficient, intercept
    
    def calculate_total_soc(long_term_soc_means):
        hex_area = 100
        total_soc = 0.0
        for long_term_soc in long_term_soc_means:
            soc_hex = long_term_soc * hex_area #soc_hex in kg
            total_soc += soc_hex
        return total_soc

    def save_biome_trends(predictions_folder, output_folder):
        biomes = ['Forest', 'Fynbos', 'Grassland', 'NamaKaroo', 'Savanna', 'SucculentKaroo', 'Thicket'] 
        biome_trend_summary = []
        biome_trend = pd.DataFrame()

        for biome in biomes:   
            biome, soc_df = trends_analysis.biome_trends(biome, output_folder, f'{predictions_folder}/combined_predictions.csv')
            annual_soc_change_percentages = soc_df['Annual_SOC_Change']
            soc_change_percentages = soc_df['SOC_Change_Percent']
            soc_long_term_means = soc_df['Mean_SOC']

            soc_long_term_means_5 = np.percentile(soc_long_term_means, 5)
            soc_long_term_means_50 = np.percentile(soc_long_term_means, 50)
            soc_long_term_means_95 = np.percentile(soc_long_term_means, 95)
           
            annual_soc_percentile_5 = np.percentile(annual_soc_change_percentages, 5)
            annual_soc_percentile_50 = np.percentile(annual_soc_change_percentages, 50)
            annual_soc_percentile_95 = np.percentile(annual_soc_change_percentages, 95)

            soc_percentile_5 = np.percentile(soc_change_percentages, 5)
            soc_percentile_50 = np.percentile(soc_change_percentages, 50)
            soc_percentile_95 = np.percentile(soc_change_percentages, 95)

            total_soc_kg = trends_analysis.calculate_total_soc(soc_long_term_means)
            
            biome_trend_summary.append([biome, total_soc_kg, soc_long_term_means_5, soc_long_term_means_50, soc_long_term_means_95 , annual_soc_percentile_5, annual_soc_percentile_50, annual_soc_percentile_95, soc_percentile_5, soc_percentile_50, soc_percentile_95])
            biome_trend = pd.concat([biome_trend, soc_df])

        biome_trend_summary_df = pd.DataFrame(biome_trend_summary, columns=['Biome', 'Total_SOC (kg)', 'SOC_Density_5_Percentile (kg/m2)', 'SOC_Density_50_Percentile (kg/m2)', 'SOC_Density_95_Percentile (kg/m2)', 'Annual_SOC_Change_5_Percentile (kg/m2)', 'Annual_SOC_Change_50_Percentile (kg/m2)', 'Annual_SOC_Change_95_Percentile (kg/m2)', 'Relative_SOC_Change_5_Percentile', 'Relative_SOC_Change_50_Percentile', 'Relative_SOC_Change_95_Percentile'])
        os.makedirs(output_folder, exist_ok=True)
        biome_trend_summary_df.to_csv(f'{output_folder}/Biome_Trends_Summary.csv', index=False)
        biome_trend.to_csv(f'{output_folder}/Biome_Trends.csv', index=False)
        biome_trend['geometry'] = [Point(xy) for xy in zip(biome_trend['Lon'], biome_trend['Lat'])]
        biome_trend = gpd.GeoDataFrame(biome_trend, geometry='geometry', crs='EPSG:4326')
        plot_utils.plot_Biome_Trends(biome_trends=biome_trend, biome_trends_col='Mean_SOC', map_output_path=f'{output_folder}/Biome_SOC.png')
        plot_utils.plot_Biome_DensityPlot(biome_trends=biome_trend, biome_trends_col='Mean_SOC', map_output_path=f'{output_folder}/Biome_SOC_Density.png')
'''
trends_analysis.save_biome_trends('MapsTrends/CNN_Model_15360/Predictions', 'MapsTrends/CNN_Model_15360/Trends')
trends_analysis.save_biome_trends('MapsTrends/CNN_Model_30720/Predictions', 'MapsTrends/CNN_Model_30720/Trends')
trends_analysis.save_biome_trends('MapsTrends/CNN_Model_61440/Predictions', 'MapsTrends/CNN_Model_61440/Trends')

trends_analysis.save_biome_trends('MapsTrends/RF_Model_15360/Predictions', 'MapsTrends/RF_Model_15360/Trends')
trends_analysis.save_biome_trends('MapsTrends/RF_Model_30720/Predictions', 'MapsTrends/RF_Model_30720/Trends')
trends_analysis.save_biome_trends('MapsTrends/RF_Model_61440/Predictions', 'MapsTrends/RF_Model_61440/Trends')
'''