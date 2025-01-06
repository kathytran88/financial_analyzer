import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle

def train_random_forest():
    df = pd.read_csv('./flask_app/credit-risk-dataset.csv')
    

    X = df[['LoanDuration', 'LoanAmount', 'InstallmentPercent', 'Age', 'ExistingCreditsCount']].values

    y = df['Risk'].apply(lambda x: 1 if x == 'Risk' else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)

    y_prediction = rf.predict(X_test)
    print(f"Accuracy on y test: {accuracy_score(y_test, y_prediction)}")

    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf, file)

train_random_forest()
