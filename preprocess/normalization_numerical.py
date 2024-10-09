import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

def normalize_numerical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    # Check for columns with more than 20% null values and remove those rows
    for column in columns:
        if column in df.columns:
            null_percentage = df[column].isnull().mean()
            if null_percentage > 0.2:
                df = df.dropna(subset=[column])
            else:
                df = df.dropna()
    
    # Create scaler instances
    scalers = {
        'min-max': MinMaxScaler(),
        'z-score': StandardScaler(),
        'max-abs': MaxAbsScaler()
    }
    
    # Check if the selected method exists in scalers
    if method not in scalers:
        raise ValueError(f"Normalization method '{method}' is not recognized.")
    
    scaler = scalers[method]
    
    # Normalize specified columns
    for column in columns:
        if column not in df.columns:
            continue
        
        # Reshape the data to fit the scaler
        df[column] = scaler.fit_transform(df[[column]])
    
    return df
