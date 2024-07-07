from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

class data_analysis:
    channel_names = {
                    'Landsat': ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI'], 
                    'Climate': ['Precipitation', 'Minimum Temp.', 'Maximum Temp.'],
                    'Terrain': ['DEM', 'Aspect', 'Slope', 'TWI']
                    }
    
    def print_pearson_coefficient(input_data_type, input_data, target_data):
        correlations = []
        size = input_data.shape[-2]
        no_of_channels = input_data.shape[-1]
        correlations = []
        
        print(f"\nCalculating Pearson's Coefficient for {input_data_type}\n")

        for i in range(no_of_channels):
            channel_flat = input_data[:,:,:,i].flatten()
            target_flat = np.repeat(target_data, size * size)

            corr, _ = pearsonr(channel_flat, target_flat)
            correlations.append(corr)
            print(f"Channel {i}: {corr}")
        
        data_analysis.scatter_plot(channel_names=data_analysis.channel_names[input_data_type], correlations=correlations, input_data_type=input_data_type)

    def scatter_plot(channel_names, correlations, input_data_type):  
        # Create scatter plots
        plt.figure(figsize=(14, 6))

        # Landsat scatter plot
        plt.subplot(1, 3, 1)
        plt.scatter(channel_names, correlations, color='blue')
        plt.title(f'{input_data_type} Channels')
        plt.xlabel('Bands')
        plt.ylabel('Pearson Coefficient')
        plt.ylim(-1, 1)    
        plt.show()    

    def heat_map(correlations, channels):
        df = pd.DataFrame({'Channel': channels, 'Correlation': correlations})
        df_sorted = df.sort_values(by='Correlation', ascending=False)


        # Plot heatmap
        plt.figure(figsize=(10, 8))
        colors = ['skyblue' if val >= 0 else 'lightcoral' for val in df_sorted['Correlation']]
        plt.barh(df_sorted['Channel'], df_sorted['Correlation'], color=colors)
       # sns.heatmap(df, annot=True, cmap='coolwarm_r', center=0, 
       #             cbar_kws={'label': f'Carbon (% by Mass)'})
        plt.xlabel('Pearson Correlation Coefficient')
        plt.title('Pearson Correlation Coefficients')
        plt.show()   

all_channels = ['Red', 'Green', 'Blue', 'NIR', 'NDVI', 'EVI', 'SAVI', 'RVI',
                'Precipitation', 'Minimum Temp.', 'Maximum Temp.',
                'DEM', 'Aspect', 'Slope', 'TWI']

all_correlations=[-0.26742420850666593,
            -0.21864576514563344,
            -0.19424686472894712,
            -0.00436183999941378,
            0.27200086770072285,
            0.1521794578182428,
            0.27200809360086564,
            0.2627487782000287,
            0.1700870516942247,
            0.07008338789177597,
            -0.0978996875578726,
            -0.011530411319920183,
            0.004594827513134436,
            0.18556567219532752,
            -0.026467760751789214]

#data_analysis.heat_map(all_correlations, all_channels)


