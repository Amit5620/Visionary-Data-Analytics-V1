import pandas as pd
# from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def select_features(df, method, label_column, feature_columns):
    """
    Select features based on the specified method.

    Parameters:
    - df: DataFrame containing the dataset
    - method: String specifying the feature selection method
    - label_column: The target label column name
    - feature_columns: List of feature column names to consider for selection

    Returns:
    - DataFrame with selected features
    """
    X = df[feature_columns]
    y = df[label_column]

    # Check for null values and remove rows with more than 20% null values
    null_threshold = 0.2
    if X.isnull().mean().max() > null_threshold:
        X.dropna(inplace=True)
        y = y[X.index]  # Align target variable with features

    # Select features based on the chosen method
    if method == "genetic-algorithm":
        # Implement a simple feature selection using Random Forest
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=feature_columns)
        selected_features = feature_importance.nlargest(10).index.tolist()  # Select top 10 features
        return df[selected_features + [label_column]]  # Return selected features and label

    elif method == "permutation":
        # Permutation importance using Random Forest
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = permutation_importance(model, X, y, n_repeats=30, random_state=0)
        feature_importance = pd.Series(importances.importances_mean, index=feature_columns)
        selected_features = feature_importance.nlargest(10).index.tolist()  # Select top 10 features
        return df[selected_features + [label_column]]  # Return selected features and label

    elif method == "xai":
        # Placeholder for XAI-based feature selection (e.g., SHAP, LIME)
        # Implement your XAI feature selection here
        pass

    return df  # If no method matched, return the original DataFrame
