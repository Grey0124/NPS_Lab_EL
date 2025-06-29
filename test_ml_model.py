import joblib
import numpy as np
import os

# Update this path if needed
MODEL_PATH = os.path.abspath("arp_app/backend/models/realistic_rf_model.joblib")

print(f"Loading model from: {MODEL_PATH}")
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

print("Model loaded successfully!")
print("Feature names:", feature_names)

# Create a dummy feature vector (all zeros, adjust length as needed)
dummy_features = np.zeros((1, len(feature_names)))

# Scale features
scaled_features = scaler.transform(dummy_features)

# Predict
prediction = model.predict(scaled_features)[0]
probabilities = model.predict_proba(scaled_features)[0]
confidence = np.max(probabilities)

# Decode prediction
if label_encoder:
    prediction_label = label_encoder.inverse_transform([prediction])[0]
else:
    prediction_label = prediction

print(f"Prediction: {prediction_label}")
print(f"Confidence: {confidence:.3f}") 