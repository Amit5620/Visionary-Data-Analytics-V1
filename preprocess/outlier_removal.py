import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, method: str, columns: list, null_threshold: float = 0.2) -> pd.DataFrame:
    """
    Removes outliers from the specified numerical columns in a DataFrame based on the selected method.
    Columns with more than null_threshold percentage of null values are removed. 
    For other columns, rows with null values are dropped before outlier removal.

    Parameters:
    - df: pd.DataFrame - The DataFrame to process.
    - method: str - The outlier removal method to apply ('iqr', 'zscore', or 'modified-zscore').
    - columns: list - The list of columns to process.
    - null_threshold: float - The threshold for null value percentage to remove columns.

    Returns:
    - pd.DataFrame - The DataFrame with outliers removed.
    """
    
    # Step 1: Check each column for null values and process accordingly
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue
        
        null_percentage = df[column].isnull().mean()
        
        if null_percentage > null_threshold:
            print(f"Removing column '{column}' due to {null_percentage * 100:.2f}% null values.")
            df.drop(column, axis=1, inplace=True)
            continue  # Skip outlier removal for this column
        
        # Step 2: Drop rows with null values for columns that are below the null threshold
        df = df.dropna(subset=[column])

        # Step 3: Remove outliers using the specified method
        if method == 'iqr':
            # IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        elif method == 'zscore':
            # Z-Score method
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores <= 3]  # Keep rows with Z-Score less than or equal to 3

        elif method == 'modified-zscore':
            # Modified Z-Score method
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))  # Median Absolute Deviation
            modified_z_scores = 0.6745 * (df[column] - median) / mad
            df = df[np.abs(modified_z_scores) <= 3.5]  # Keep rows with modified Z-Score <= 3.5

        else:
            print(f"Warning: Outlier removal method '{method}' is not recognized. No changes made to column '{column}'.")

    return df
