import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle

def train_decision_tree():
    df = pd.read_csv('./flask_app/credit-risk-dataset.csv')
    
    X = df[['LoanDuration', 'LoanAmount', 'InstallmentPercent', 'Age', 'ExistingCreditsCount']].values
    
    y = df['Risk'].apply(lambda x: 1 if x == 'Risk' else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    creditTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    creditTree.fit(X_train, y_train)
    
    y_prediction = creditTree.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_prediction)}")

    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(creditTree, file)

train_decision_tree()
