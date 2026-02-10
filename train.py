import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

#----- Version handling ----
MODEL_VERSION = sys.argv[1] if len(sys.argv) > 1 else 'v1'

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/model_{MODEL_VERSION}.joblib"

data = pd.read_csv('data/data.csv')

X = data[['age','salary']]
y = data['bought']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

print(f"Model version {MODEL_VERSION} trained and saved at {MODEL_PATH}")