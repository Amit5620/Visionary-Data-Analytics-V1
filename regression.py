from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import math
from sklearn.model_selection import train_test_split

def regression_prediction(df, predicted_column):
    X = df.drop(predicted_column, axis=1)
    y = df[predicted_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Support Vector Regressor': SVR(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'XGBoost Regressor': XGBRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        }

    return results