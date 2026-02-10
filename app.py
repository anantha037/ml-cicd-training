from fastapi import FastAPI
import joblib
import numpy as np
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_FAILURES = Counter(
    "prediction_failure_total",
    "Total number of failed prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction"
)

app = FastAPI()

ACTIVE_MODEL_VERSION = os.getenv("MODEL_VERSION","v1")
MODEL_PATH = f"models/model_{ACTIVE_MODEL_VERSION}.joblib"
logger.info(f"Loading model version: {ACTIVE_MODEL_VERSION}")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")

@app.get("/")
def home():
    return {"message":"ML Model API is running"}


@app.post("/predict")
def predict(age:int, salary:int):
    start_time = time.time()
    REQUEST_COUNT.inc()


    try:
        logger.info(f"Received prediction request: age={age}, salary={salary}")


        input_data = np.array([[age, salary]])
        prediction = model.predict(input_data)

        result = int(prediction[0])
        logger.info(f"Prediction result: {result}")
        return {
            "age":age,
            "salary":salary,
            "prediction":result
        }

    except Exception as e:
        REQUEST_FAILURES.inc()
        logger.error(f"Prediction failed: {str(e)}")
        return {"error":"Prediction failed"}

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")