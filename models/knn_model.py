import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_knn():
    df = pd.read_csv('./flask_app/credit-risk-dataset.csv')
    
    X = df[['LoanDuration', 'LoanAmount', 'InstallmentPercent', 'Age', 'ExistingCreditsCount']].values
    
    y = df['Risk'].apply(lambda x: 1 if x == 'Risk' else 0).values

    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    
    yhat = knn.predict(X_test)
    print(f'Train set accuracy: {accuracy_score(y_train, knn.predict(X_train))}')
    print(f'Test set accuracy: {accuracy_score(y_test, yhat)}')
    
    with open('knn_model.pkl', 'wb') as file:
        pickle.dump(knn, file)

train_knn()
