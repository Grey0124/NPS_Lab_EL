#!/usr/bin/env python3
"""
Test script for the trained IsolationForest ARP spoofing detection model.
This script loads the saved model and demonstrates how to use it for predictions.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

def load_model(model_path='models/iforest.joblib'):
    """Load the trained IsolationForest model."""
    print(f"Loading model from: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        
        # Extract components
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        contamination = model_data['contamination']
        training_date = model_data['training_date']
        model_info = model_data['model_info']
        
        print("✅ Model loaded successfully!")
        print(f"📅 Training date: {training_date}")
        print(f"🔧 Contamination: {contamination}")
        print(f"📊 Model info: {model_info}")
        print(f"🔍 Features: {len(feature_names)}")
        
        return model, scaler, feature_names, model_data
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

def predict_anomaly(model, scaler, feature_names, sample_data):
    """
    Predict if a sample is anomalous.
    
    Args:
        model: Trained IsolationForest model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        sample_data: Sample data to predict (numpy array or list)
    
    Returns:
        dict: Prediction results
    """
    try:
        # Convert to numpy array if needed
        if isinstance(sample_data, list):
            sample_data = np.array(sample_data).reshape(1, -1)
        elif len(sample_data.shape) == 1:
            sample_data = sample_data.reshape(1, -1)
        
        # Check feature count
        if sample_data.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {sample_data.shape[1]}")
        
        # Scale the data
        sample_scaled = scaler.transform(sample_data)
        
        # Get prediction and score
        prediction = model.predict(sample_scaled)[0]
        anomaly_score = model.decision_function(sample_scaled)[0]
        
        # Convert prediction: -1 for anomaly, 1 for normal
        is_anomaly = prediction == -1
        
        return {
            'prediction': prediction,
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'confidence': abs(anomaly_score)  # Higher absolute score = more confident
        }
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return None

def test_with_sample_data():
    """Test the model with sample ARP packet data."""
    print("\n" + "="*50)
    print("TESTING MODEL WITH SAMPLE DATA")
    print("="*50)
    
    # Load model
    model, scaler, feature_names, model_data = load_model()
    
    if model is None:
        print("❌ Cannot proceed without loaded model")
        return
    
    # Create sample data (normal ARP packet)
    print(f"\n📋 Feature names: {feature_names}")
    
    # Sample 1: Normal ARP packet (all zeros as baseline)
    normal_sample = np.zeros(len(feature_names))
    print(f"\n🔍 Testing normal sample: {normal_sample[:5]}...")
    
    result1 = predict_anomaly(model, scaler, feature_names, normal_sample)
    if result1:
        print(f"   Prediction: {'ANOMALY' if result1['is_anomaly'] else 'NORMAL'}")
        print(f"   Anomaly Score: {result1['anomaly_score']:.4f}")
        print(f"   Confidence: {result1['confidence']:.4f}")
    
    # Sample 2: Slightly anomalous (some non-zero values)
    anomalous_sample = np.zeros(len(feature_names))
    anomalous_sample[0] = 1.0  # Modify first feature
    anomalous_sample[2] = -0.5  # Modify third feature
    print(f"\n🔍 Testing anomalous sample: {anomalous_sample[:5]}...")
    
    result2 = predict_anomaly(model, scaler, feature_names, anomalous_sample)
    if result2:
        print(f"   Prediction: {'ANOMALY' if result2['is_anomaly'] else 'NORMAL'}")
        print(f"   Anomaly Score: {result2['anomaly_score']:.4f}")
        print(f"   Confidence: {result2['confidence']:.4f}")
    
    # Sample 3: Highly anomalous (extreme values)
    extreme_sample = np.zeros(len(feature_names))
    extreme_sample[0] = 10.0  # Very high value
    extreme_sample[1] = -10.0  # Very low value
    print(f"\n🔍 Testing extreme sample: {extreme_sample[:5]}...")
    
    result3 = predict_anomaly(model, scaler, feature_names, extreme_sample)
    if result3:
        print(f"   Prediction: {'ANOMALY' if result3['is_anomaly'] else 'NORMAL'}")
        print(f"   Anomaly Score: {result3['anomaly_score']:.4f}")
        print(f"   Confidence: {result3['confidence']:.4f}")

def test_with_real_data_sample():
    """Test with a small sample from the real dataset."""
    print("\n" + "="*50)
    print("TESTING MODEL WITH REAL DATA SAMPLE")
    print("="*50)
    
    # Load model
    model, scaler, feature_names, model_data = load_model()
    
    if model is None:
        print("❌ Cannot proceed without loaded model")
        return
    
    try:
        # Load a small sample from the dataset
        print("📊 Loading sample from dataset...")
        df_sample = pd.read_csv('../dataset.csv', nrows=10)
        
        # Remove label column if present
        if 'label' in df_sample.columns:
            df_sample = df_sample.drop('label', axis=1)
        
        print(f"📋 Sample shape: {df_sample.shape}")
        print(f"📋 Sample columns: {list(df_sample.columns)}")
        
        # Test each sample
        for i in range(min(5, len(df_sample))):
            sample = df_sample.iloc[i].values
            print(f"\n🔍 Testing real sample {i+1}: {sample[:5]}...")
            
            result = predict_anomaly(model, scaler, feature_names, sample)
            if result:
                print(f"   Prediction: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
                print(f"   Anomaly Score: {result['anomaly_score']:.4f}")
                print(f"   Confidence: {result['confidence']:.4f}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {str(e)}")

def main():
    """Main test function."""
    print("IsolationForest ARP Spoofing Detection - Model Testing")
    print("="*60)
    
    # Test model loading
    test_with_sample_data()
    
    # Test with real data sample
    test_with_real_data_sample()
    
    print("\n" + "="*60)
    print("✅ Model testing completed!")
    print("="*60)

if __name__ == "__main__":
    main() 