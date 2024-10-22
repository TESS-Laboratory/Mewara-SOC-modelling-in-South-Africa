import os
from DataProcessing import grid_utils

class world_clim:
    def clip_world_clim():
        world_clim_path = r'Data/WorldClim'
        world_clim_output_path = r'Data/WorldClim_SA'
        south_africa = grid_utils.grid_utils.get_sa_shape()

        for file in os.listdir(world_clim_path):
            filename = os.fsdecode(file)
            if (filename.endswith('.tif')):
                grid_utils.grid_utils.clip_to_sa(rasterfile_path=os.path.join(world_clim_path, filename), 
                                      south_africa=south_africa, 
                                      output_path=os.path.join(world_clim_output_path, filename))
