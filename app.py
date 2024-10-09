from flask import Flask, render_template, request, redirect, url_for
from flask import render_template_string
from analysis import dataset_analysis
from preprocessing import data_preprocessing
from prediction import make_prediction
import pandas as pd
import os
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        file = request.files['file']
        analysis_result = dataset_analysis(file)
        
        return render_template_string(analysis_result)
        # return render_template('analysis-result.html', result=analysis_result)
    return render_template('analysis.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['dataset']
        if not file:
            return render_template('preprocess.html', error="Please upload a file")

        # Extract all 10 inputs from the form
        label_column = request.form['prediction-column']  # 1. Prediction column name
        cleaning_numerical = request.form['cleaning-numerical']  # 2. Data Cleaning (Numerical Features)
        cleaning_numerical_columns = request.form['cleaning-numerical-columns']  # 3. Numerical Cleaning Columns
        cleaning_categorical = request.form['cleaning-categorical']  # 4. Data Cleaning (Categorical Features)
        cleaning_categorical_columns = request.form['cleaning-categorical-columns']  # 5. Categorical Cleaning Columns
        outlier_removal = request.form['outlier-removal']  # 6. Outlier Removal
        outlier_removal_columns = request.form['outlier-removal-columns']  # 7. Outlier Removal Columns
        transformation_numerical = request.form['transformation-numerical']  # 8. Data Transformation (Numerical)
        transformation_numerical_columns = request.form['transformation-numerical-columns']  # 9. Numerical Transformation Columns
        normalization_numerical = request.form['normalization-numerical']  # 10. Data Normalization (Numerical)
        normalization_numerical_columns = request.form['normalization-numerical-columns']  # 11. Normalization Columns
        encoding_categorical = request.form['encoding-categorical']  # 12. Data Encoding (Categorical)
        encoding_categorical_columns = request.form['encoding-categorical-columns']  # 13. Encoding Columns
        transformation_categorical = request.form['transformation-categorical']  # 14. Data Transformation (Categorical)
        transformation_categorical_columns = request.form['transformation-categorical-columns']  # 15. Categorical Transformation Columns
        feature_selection = request.form['feature-selection']  # 16. Feature Selection
        feature_selection_columns = request.form['feature-selection-columns']  # 17. Feature Selection Columns

        # Convert the uploaded file to a DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template('preprocess.html', error="Error reading the CSV file. Please upload a valid CSV file.")

        # Print extracted values for debugging purposes
        print(label_column, cleaning_numerical, cleaning_numerical_columns,
              cleaning_categorical, cleaning_categorical_columns, outlier_removal,
              outlier_removal_columns, transformation_numerical, transformation_numerical_columns,
              normalization_numerical, normalization_numerical_columns, encoding_categorical,
              encoding_categorical_columns, transformation_categorical, transformation_categorical_columns,
              feature_selection, feature_selection_columns)
        
        # Call your data preprocessing function
        processed_data = data_preprocessing(df, label_column, cleaning_numerical, cleaning_numerical_columns, 
                                     cleaning_categorical, cleaning_categorical_columns, outlier_removal, 
                                     outlier_removal_columns, transformation_numerical, transformation_numerical_columns, 
                                     normalization_numerical, normalization_numerical_columns, encoding_categorical, 
                                     encoding_categorical_columns, transformation_categorical, transformation_categorical_columns, 
                                     feature_selection, feature_selection_columns, 'output/preprocessed_data.csv')

        # Save the processed data as a CSV file in the static directory
        csv_filename = 'processed_data.csv'
        csv_path = os.path.join('output', csv_filename)
        processed_data.to_csv(csv_path, index=False)

         # Display the processed data
        return render_template('preprocess-result.html', result=processed_data.to_html(classes="table table-bordered table-striped"), csv_filename=csv_filename)

    return render_template('preprocess.html')

@app.route('/download/<filename>')
def download_file(filename):
    # Provide the path to the static directory and the filename
    return send_from_directory('output', filename, as_attachment=True)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['dataset-upload']
        df = pd.read_csv(file)
        predicted_column = request.form['prediction-column']
        problem_type = request.form.getlist('problem-type')

        if 'regression' in problem_type:
            from regression import regression_prediction
            results = regression_prediction(df, predicted_column)
            return render_template('prediction-result.html', results=results, problem_type='Regression')

        elif 'classification' in problem_type:
            from classification import classification_prediction
            results = classification_prediction(df, predicted_column)
            return render_template('prediction-result.html', results=results, problem_type='Classification')

        else:
            return render_template('prediction-result.html', error='Please select a problem type')

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
