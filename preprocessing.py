import pandas as pd
from preprocess.cleaning_numerical import clean_numerical
from preprocess.cleaning_categorical import clean_categorical
from preprocess.outlier_removal import remove_outliers
from preprocess.transformation_numerical import transform_numerical
from preprocess.normalization_numerical import normalize_numerical
from preprocess.encoding_categorical import encode_categorical
from preprocess.transformation_categorical import transform_categorical


def data_preprocessing(df, label_column, cleaning_numerical, cleaning_numerical_columns, 
                       cleaning_categorical, cleaning_categorical_columns, outlier_removal, 
                       outlier_removal_columns, transformation_numerical, transformation_numerical_columns, 
                       normalization_numerical, normalization_numerical_columns, encoding_categorical, 
                       encoding_categorical_columns, transformation_categorical, transformation_categorical_columns, 
                       feature_selection, feature_selection_columns, output_file):

    # Step 1: Clean numerical data
    numerical_columns = cleaning_numerical_columns.split(',')
    df = clean_numerical(df, cleaning_numerical, numerical_columns)

    # Step 2: Clean categorical data
    categorical_columns = cleaning_categorical_columns.split(',')
    df = clean_categorical(df, cleaning_categorical, categorical_columns)

    # Step 3: Remove outliers
    outlier_columns = outlier_removal_columns.split(',')
    df = remove_outliers(df, outlier_removal, outlier_columns)

    # Step 4: Transform numerical data
    transformation_numerical_columns = transformation_numerical_columns.split(',')
    df = transform_numerical(df, transformation_numerical, transformation_numerical_columns)

    # Step 5: Normalize numerical data
    normalization_numerical_columns = normalization_numerical_columns.split(',')
    df = normalize_numerical(df, normalization_numerical, normalization_numerical_columns)

    # Step 6: Encode categorical data
    encoding_categorical_columns = encoding_categorical_columns.split(',')
    df = encode_categorical(df, encoding_categorical, encoding_categorical_columns)

    # Step 7: Transform categorical data
    transformation_categorical_columns = transformation_categorical_columns.split(',')
    df = transform_categorical(df, transformation_categorical, transformation_categorical_columns)

    # Step 8: Feature selection
    # feature_selection_columns = feature_selection_columns.split(',')
    # if feature_selection_columns != 'na':
    #     df = select_features(df, feature_selection, label_column, feature_selection_columns)

    # Step 9: Save the preprocessed DataFrame to a CSV file
    df.to_csv(output_file, index=False)  # Save without the index

    return df
