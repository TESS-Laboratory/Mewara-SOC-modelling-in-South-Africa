
import pandas as pd
from grid_utils import grid_utils

class grid:
    @staticmethod
    def save_soc_square_grid():
        soil_data = pd.read_csv(r'DataProcessing/soc_gdf.csv')
        square_grid = grid_utils.get_square_grid(grid_size_in_meters=30720)
        carbon_mapping = grid_utils.get_carbon_mapping()
        joined = grid_utils.get_square_grid_avg_carbon_color(square_grid=square_grid,
                                                             carbon_mapping=carbon_mapping,
                                                             soil_data=soil_data) 
        square_grid.to_csv(r'DataProcessing/square_grid.csv')
        joined.to_csv(r'DataProcessing/soc_square_grid.csv')
        grid_utils.plot_heat_map(soil_data_with_avg_c_color=joined, 
                                 title=f'Long Term Average Carbon (% by Mass) of South Africa (1986-2022)')
        #input('Press any key to continue')

    @staticmethod
    def save_soc_hex_grid():
        soil_data = pd.read_csv(r'DataProcessing/soc_gdf.csv')
        hex_grid = grid_utils.get_hex_grid(hex_size=0.1)
        carbon_mapping = grid_utils.get_carbon_mapping()
        joined = grid_utils.get_hex_grid_avg_carbon_color(hex_grid=hex_grid,
                                                          carbon_mapping=carbon_mapping,
                                                          soil_data=soil_data) 
        hex_grid.to_csv(r'DataProcessing/hex_grid.csv')
        joined.to_csv(r'DataProcessing/soc_hex_grid.csv')
        grid_utils.plot_heat_map(soil_data_with_avg_c_color=joined, 
                                 title=f'Long Term Average Carbon (% by Mass) of South Africa (1986-2022)')
        input('Press any key to continue')

grid.save_soc_hex_grid()
