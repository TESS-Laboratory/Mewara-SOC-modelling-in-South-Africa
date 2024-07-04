import os
from DataProcessing.grid_utils import grid_utils

for year in range(2008, 2009):
    landsat_path = f'Data\LandSat\Annual_Processed\Landsat_{year}.tif'
    if os.path.exists(landsat_path):
        grid_utils.clip_to_sa(rasterfile_path=landsat_path, 
                            south_africa=grid_utils.get_sa_shape(), 
                            output_path=f'Data\LandSat\Annual_Processed\Clipped_Landsat_{year}.tif')

