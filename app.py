from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter('student_api_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('student_api_http_request_duration_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('student_api_predictions_total', 'Total predictions', ['result'])
MODEL_LOADED = Gauge('student_api_model_loaded', 'Model loaded status')

MODEL_LOADED.set(1)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if request.path != '/metrics':
        latency = time.time() - request.start_time
        REQUEST_LATENCY.labels(endpoint=request.path).observe(latency)
        REQUEST_COUNT.labels(method=request.method, endpoint=request.path, status=response.status_code).inc()
    return response

MODEL_PATH = Path("student_result_model.pkl")
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("student_result_model.pkl was not found. Run training first.")
    return joblib.load(MODEL_PATH)

@app.route("/")
def health_check():
    return jsonify({"service": "student-result-prediction", "status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    data = request.get_json()
    df = pd.DataFrame([data], columns=FEATURES)
    prediction = int(model.predict(df)[0])
    result_text = "PASS" if prediction == 1 else "FAIL"
    return jsonify({"prediction": result_text, "prediction_code": prediction})

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
