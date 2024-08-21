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
    
    def get_predictions_by_biome(biome, predictions_path):
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

        return biome_predictions
    
    def biome_trends(biome, predictions_path):
        biome_predictions = trends_analysis.get_predictions_by_biome(biome=biome, predictions_path=predictions_path)
        
        return trends_analysis.get_SOC_trend(biome, biome_predictions)

    def get_SOC_trend(biome, biome_predictions):
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
        return np.sum(long_term_soc_means) * hex_area

    def soc_long_term_percentiles(soc_df):
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
        return soc_long_term_means_5,soc_long_term_means_50,soc_long_term_means_95,annual_soc_percentile_5,annual_soc_percentile_50,annual_soc_percentile_95,soc_percentile_5,soc_percentile_50,soc_percentile_95,total_soc_kg
    
    def save_biome_trends(predictions_folder, output_folder):
        biomes = ['Forest', 'Fynbos', 'Grassland', 'NamaKaroo', 'Savanna', 'SucculentKaroo', 'Thicket'] 
        biome_trend_summary = []
        biome_trend = pd.DataFrame()
        combined_predictions_path = f'{predictions_folder}/combined_predictions.csv'
        predictions = pd.read_csv(combined_predictions_path)
        predictions = predictions[(predictions['Year'] >= 2000) & (predictions['Year'] <= 2023)]

        for biome in biomes:   
            biome, soc_df = trends_analysis.biome_trends(biome, combined_predictions_path)

            if soc_df.empty:
                continue
            
            soc_long_term_means_5, soc_long_term_means_50, soc_long_term_means_95, annual_soc_percentile_5, annual_soc_percentile_50, annual_soc_percentile_95, soc_percentile_5, soc_percentile_50, soc_percentile_95, biome_total_soc_kg = trends_analysis.soc_long_term_percentiles(soc_df)
            
            biome_trend_summary.append([biome, biome_total_soc_kg, soc_long_term_means_5, soc_long_term_means_50, soc_long_term_means_95 , annual_soc_percentile_5, annual_soc_percentile_50, annual_soc_percentile_95, soc_percentile_5, soc_percentile_50, soc_percentile_95])
            biome_trend = pd.concat([biome_trend, soc_df])

        total_long_term_means_5, total_long_term_means_50, total_long_term_means_95, annual_total_percentile_5, annual_total_percentile_50, annual_total_percentile_95, total_percentile_5, total_percentile_50, total_percentile_95, total_soc_kg = trends_analysis.soc_long_term_percentiles(biome_trend)
        biome_trend_summary.append(['Total', total_soc_kg, total_long_term_means_5, total_long_term_means_50, total_long_term_means_95 , annual_total_percentile_5, annual_total_percentile_50, annual_total_percentile_95, total_percentile_5, total_percentile_50, total_percentile_95])

        biome_trend_summary_df = pd.DataFrame(biome_trend_summary, columns=['Biome', 'Total_SOC (kg)', 'SOC_Density_5_Percentile (kg/m2)', 'SOC_Density_50_Percentile (kg/m2)', 'SOC_Density_95_Percentile (kg/m2)', 'Annual_SOC_Change_5_Percentile (kg/m2)', 'Annual_SOC_Change_50_Percentile (kg/m2)', 'Annual_SOC_Change_95_Percentile (kg/m2)', 'Relative_SOC_Change_5_Percentile', 'Relative_SOC_Change_50_Percentile', 'Relative_SOC_Change_95_Percentile'])
        os.makedirs(output_folder, exist_ok=True)
        biome_trend_summary_df.to_csv(f'{output_folder}/Biome_Trends_Summary.csv', index=False)
        biome_trend.to_csv(f'{output_folder}/Biome_Trends.csv', index=False)
        biome_trend['geometry'] = [Point(xy) for xy in zip(biome_trend['Lon'], biome_trend['Lat'])]
        biome_trend = gpd.GeoDataFrame(biome_trend, geometry='geometry', crs='EPSG:4326')
        plot_utils.plot_Biome_Trends(biome_trends=biome_trend, biome_trends_col='Mean_SOC', map_output_path=f'{output_folder}/Biome_SOC.png')
        plot_utils.plot_Biome_DensityPlot(biome_trends=biome_trend, biome_trends_col='Mean_SOC', map_output_path=f'{output_folder}/Biome_SOC_Density.png')
