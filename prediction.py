import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def make_prediction(file, label_column, algorithms):
    # Load dataset
    df = pd.read_csv(file)

    # Split dataset into features and target
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Initialize results dictionary
    results = {}

    # Iterate over selected algorithms
    for algorithm in algorithms:
        if algorithm == 'Linear Regression':
            model = LinearRegression()
        elif algorithm == 'Logistic Regression':
            model = LogisticRegression()
        elif algorithm == 'KNN':
            model = KNeighborsClassifier()
        elif algorithm == 'SVM':
            model = SVC()
        elif algorithm == 'Naive Bayes':
            model = GaussianNB()
        elif algorithm == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif algorithm == 'Random Forest':
            model = RandomForestClassifier()
        # elif algorithm == 'XGBoost':
        #     model = XGBClassifier()

        # Train model
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        # Store results
        results[algorithm] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results