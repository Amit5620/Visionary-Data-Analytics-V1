import pandas as pd
from sklearn.impute import KNNImputer

def clean_numerical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    """
    Cleans numerical columns in a DataFrame based on the specified method.
    Removes columns with more than 20% null values.

    Parameters:
    - df: pd.DataFrame - The DataFrame to clean.
    - method: str - The cleaning method to apply.
    - columns: list - The list of columns to clean.

    Returns:
    - pd.DataFrame - The cleaned DataFrame.
    """
    
    # Threshold for removing columns based on null value percentage
    null_threshold = 0.2

    # Step 1: Remove columns with more than 20% null values
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue
        
        null_percentage = df[column].isnull().mean()
        
        if null_percentage > null_threshold:
            print(f"Removing column '{column}' due to {null_percentage * 100:.2f}% null values.")
            df.drop(column, axis=1, inplace=True)
            continue  # Skip the cleaning process for this column

        # Step 2: Clean numerical data using the specified method
        if method == 'remove':
            # Remove rows with NaN values in specified columns
            df = df.dropna(subset=[column])

        elif method == 'constant':
            # Impute missing values with a constant (0 in this case)
            df[column].fillna(0, inplace=True)

        elif method == 'mean':
            # Impute missing values with the mean of the column
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

        elif method == 'median':
            # Impute missing values with the median of the column
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)

        elif method == 'mode':
            # Impute missing values with the mode of the column
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

        elif method == 'knn':
            # Use KNN imputation for missing values
            knn_imputer = KNNImputer(n_neighbors=5)
            df[column] = knn_imputer.fit_transform(df[[column]])

        else:
            print(f"Warning: Cleaning method '{method}' is not recognized. No changes made to column '{column}'.")

    return df
