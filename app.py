from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()

model = joblib.load('model.joblib')

@app.get("/")
def home():
    return {"message":"ML Model API is running"}


@app.post("/predict")
def predict(age:int, salary:int):
    input_data = np.array([[age, salary]])
    prediction = model.predict(input_data)

    return {
        "age":age,
        "salary":salary,
        "prediction":int(prediction[0])
    }