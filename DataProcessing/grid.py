
import pandas as pd
from grid_utils import grid_utils

class grid:
    @staticmethod
    def save_soc_hex_grid():
        soil_data = pd.read_csv(r'DataProcessing/soc_gdf.csv')
        hex_grid = pd.read_csv(r'DataProcessing/hex_grid.csv')
        soc_hex_grid = grid_utils.get_soc_hex_grid(hex_grid_df=hex_grid,
                                             soil_data=soil_data) 
        soc_hex_grid.to_csv(r'DataProcessing/soc_hex_grid.csv', mode='w')
        avg_cols = ['Hex_ID']
      
        soc_hex_avg_c = grid_utils.get_avg_c_each_grid(df=soc_hex_grid,
                                                        group_by_cols=avg_cols,
                                                        avg_col='C',
                                                        avgc_col_rename='Avg_C')
        
        hex_grid_avg_c = pd.merge(hex_grid, soc_hex_avg_c, on='Hex_ID', how='left')
        
        grid_utils.plot_heat_map(data_avg_c_color_geometry=hex_grid_avg_c, 
                                title=f'Long Term Average Carbon (% by Mass) of South Africa (1986-2022)',
                                geometry_col='geometry',
                                savePlot=True,
                                output_plot_path='Plots/C_HeatMap.png')

grid.save_soc_hex_grid()
