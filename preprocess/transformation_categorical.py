import pandas as pd
from category_encoders import BinaryEncoder

def transform_categorical(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
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
        
        if method == 'binary':  # Binary Encoding
            be = BinaryEncoder()
            encoded_df = be.fit_transform(df[column])
            df = df.join(encoded_df).drop(column, axis=1)  # Drop the original column
        
        elif method == 'embeddings':  # Embedding Transformation (this requires a specific setup)
            # Here, you can implement logic for creating embeddings if you have a predefined model
            # For the sake of simplicity, we won't implement embeddings here
            raise NotImplementedError("Embedding transformation is not implemented. Please implement your logic.")

    return df
