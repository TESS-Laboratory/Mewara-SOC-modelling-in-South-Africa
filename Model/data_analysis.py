from scipy.stats import pearsonr
class data_analysis:
    def print_pearson_coefficient(input_data, target_data):
        target_flat = target_data.flatten()
        correlations = []

        for i in range(input_data.shape[-1]):
            channel_flat = input_data[:,:,i].flatten()
            corr, _ = pearsonr(channel_flat, target_flat)
            correlations.append(corr)

        for i, corr in enumerate(correlations):
            print(f"Pearson correlation for channel {i} against carbon as target: {corr}")


