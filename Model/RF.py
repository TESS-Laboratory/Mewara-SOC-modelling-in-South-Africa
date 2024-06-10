from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

class RF:

    def train(features, targets):
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Root Mean Squared Error: {rmse}")
        print(f"R^2 Score: {r2}")
