import pandas as pd
import numpy as np

def transform_numerical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    # Check for columns with more than 20% null values and remove those rows
    for column in columns:
        if column in df.columns:
            null_percentage = df[column].isnull().mean()
            if null_percentage > 0.2:
                df = df.dropna(subset=[column])
            else:
                df = df.dropna()
    
    # Apply transformation based on the selected method
    for column in columns:
        if column not in df.columns:
            continue
        
        if method == 'log':
            df[column] = np.log1p(df[column])  # log1p for log(1 + x)
        elif method == 'box-cox':
            from scipy import stats
            df[column], _ = stats.boxcox(df[column].dropna())  # Box-Cox Transformation
        elif method == 'yeo-johnson':
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson')
            df[column] = pt.fit_transform(df[[column]])
    
    return df
