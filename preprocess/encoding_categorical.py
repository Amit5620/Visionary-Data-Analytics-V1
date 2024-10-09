import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

def encode_categorical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    # Check for columns with more than 20% null values and remove those rows
    for column in columns:
        if column in df.columns:
            null_percentage = df[column].isnull().mean()
            if null_percentage > 0.2:
                df = df.dropna(subset=[column])
            else:
                df = df.dropna()

    # Apply encoding based on the selected method
    for column in columns:
        if column not in df.columns:
            continue
        
        if method == 'ohe':  # One-Hot Encoding
            ohe = OneHotEncoder(sparse=False, drop='first')
            encoded_columns = ohe.fit_transform(df[[column]])
            df = df.join(pd.DataFrame(encoded_columns, columns=ohe.get_feature_names_out([column]), index=df.index))
            df = df.drop(column, axis=1)  # Drop the original column
        
        elif method == 'label':  # Label Encoding
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
        
        elif method == 'ordinal':  # Ordinal Encoding
            oe = OrdinalEncoder()
            df[column] = oe.fit_transform(df[[column]])

    return df
