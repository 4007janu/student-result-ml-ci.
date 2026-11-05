from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
