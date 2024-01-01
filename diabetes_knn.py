import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

file_path='diabetes_012_health_indicators_BRFSS2015.csv'
data=pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

X = data.iloc[:, :-1]
y = data['Diabetes_012'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 3

knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"k-NN (k={k}) Accuracy: {accuracy_knn:.2f}")

# Display classification report
print("\nk-NN Classification Report:")
print(classification_report(y_test, y_pred_knn))