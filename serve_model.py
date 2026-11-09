"""Optional tiny inference service. NOT for medical or clinical use."""
import joblib
from flask import Flask, jsonify, request
from ml_config import ROOT, FEATURES

app = Flask(__name__)
model = joblib.load(ROOT / 'models/candidate.joblib')


@app.get('/health')
def health():
    return jsonify(status='ok')


@app.post('/predict')
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or any(k not in payload for k in FEATURES):
        return jsonify(error='Provide all five numerical measurements.'), 400
    try:
        values = [float(payload[k]) for k in FEATURES]
    except (ValueError, TypeError):
        return jsonify(error='Measurements must be numeric.'), 400
    if any(not (0 <= x < float('inf')) for x in values):
        return jsonify(error='Measurements must be finite and nonnegative.'), 400
    import pandas as pd
    X = pd.DataFrame([values], columns=FEATURES)
    probability = float(model.predict_proba(X)[0, 1])
    return jsonify(malignant_probability=probability, label=int(probability >= 0.5))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
