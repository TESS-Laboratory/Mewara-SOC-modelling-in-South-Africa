import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import TheilSenRegressor

class regression_analysis:
    def theil_regression_analysis(model_name, predictions_folder):
        for year in range(1986, 2023):
            predictions_path = f'{predictions_folder}/predictions_{year}.csv'
           
            if os.path.exists(predictions_path):
                pred = pd.read_csv(predictions_path)
                df = pd.concat([df, pred], ignore_index=True)
                
                X = df[['Year', 'Lat', 'Lon']]
                y = df['C']

                # Initialize and fit the Theil-Sen regressor
                theil_sen = TheilSenRegressor()
                theil_sen.fit(X, y)

                coefficients = theil_sen.coef_
                intercept = theil_sen.intercept_

                coefficients, intercept

    def plot_theil_regression():
        plt.figure(figsize=(10, 6))

        # Plot coefficients
        plt.bar(features, coefficients, color='blue', label='Coefficients')
        plt.axhline(y=0, color='black', linewidth=0.5)

        # Plot intercept
        plt.scatter(['Intercept'], [intercept], color='red', zorder=5, label='Intercept')

        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.title('Theil-Sen Regression Coefficients and Intercept')
        plt.legend()
        plt.show()
               
#regression_analysis.theil_regression_analysis(model_name='CNN', predictions_folder='Maps/CNN_Model.keras/Predictions')