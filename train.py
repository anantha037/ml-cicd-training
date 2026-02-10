import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('data/data.csv')

X = data[['age','salary']]
y = data['bought']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'model.joblib')

print("Model training completed successfully")