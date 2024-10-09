import pandas as pd

def clean_categorical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    """
    Cleans categorical columns in a DataFrame based on the specified method.
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

        # Step 2: Clean categorical data using the specified method
        if method == 'remove':
            # Remove rows with NaN values in specified columns
            df = df.dropna(subset=[column])

        elif method == 'constant':
            # Impute missing values with a constant (e.g., 'Unknown')
            df[column].fillna('Unknown', inplace=True)

        elif method == 'mode':
            # Impute missing values with the mode of the column
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

        else:
            print(f"Warning: Cleaning method '{method}' is not recognized. No changes made to column '{column}'.")

    return df
