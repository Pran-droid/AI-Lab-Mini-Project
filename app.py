import math
import pickle

import numpy as np
from flask import Flask, flash, render_template, request

app = Flask(__name__)
app.secret_key = "change-me-in-production"

MODEL_PATH = "credit.pkl"
FEATURE_ORDER = [
    "time",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",
    "amount",
]

SIMPLE_FIELDS = ["time", "amount", "v1", "v2", "v3", "v4", "v10", "v14"]

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def parse_float(form, key):
    value = form.get(key, "").strip()
    if value == "":
        raise ValueError(f"{key} is required")
    return float(value)


def build_feature_vector(values):
    filled = {name: 0.0 for name in FEATURE_ORDER}
    filled.update(values)
    return np.array([[filled[name] for name in FEATURE_ORDER]], dtype=float)


def predict_with_confidence(features):
    prediction = int(model.predict(features)[0])
    confidence = None
    fraud_score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        if hasattr(model, "classes_") and 1 in model.classes_:
            fraud_index = int(list(model.classes_).index(1))
        else:
            fraud_index = 1 if len(probabilities) > 1 else 0
        fraud_score = float(probabilities[fraud_index])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        fraud_score = 1.0 / (1.0 + math.exp(-score))

    if fraud_score is not None:
        confidence = fraud_score if prediction == 1 else (1.0 - fraud_score)

    return prediction, confidence, fraud_score


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {name: parse_float(request.form, name) for name in SIMPLE_FIELDS}
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("index.html", form_values=request.form)

    features = build_feature_vector(values)
    prediction, confidence, fraud_score = predict_with_confidence(features)

    result = {
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "is_fraud": prediction == 1,
        "confidence": confidence,
        "fraud_score": fraud_score,
    }

    return render_template("index.html", result=result, form_values=values)


if __name__ == "__main__":
    app.run(debug=True)
