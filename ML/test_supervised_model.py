#!/usr/bin/env python3
"""
Test script for the trained RandomForest supervised ARP spoofing detection model.
This script loads the saved model and demonstrates how to use it for predictions.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

def load_model(model_path='models/rf_model.joblib'):
    """Load the trained RandomForest model."""
    print(f"Loading model from: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        
        # Extract components
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        class_names = model_data['class_names']
        training_date = model_data['training_date']
        model_info = model_data['model_info']
        
        print("✅ Model loaded successfully!")
        print(f"📅 Training date: {training_date}")
        print(f"📊 Model info: {model_info}")
        print(f"🔍 Features: {len(feature_names)}")
        print(f"🎯 Classes: {len(class_names)}")
        print(f"🏷️  Class names: {class_names}")
        
        return model, scaler, label_encoder, feature_names, class_names, model_data
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None, None

def predict_class(model, scaler, label_encoder, feature_names, class_names, sample_data):
    """
    Predict the class of a sample.
    
    Args:
        model: Trained RandomForest model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature names
        class_names: List of class names
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
        
        # Get prediction and probabilities
        prediction_encoded = model.predict(sample_scaled)[0]
        probabilities = model.predict_proba(sample_scaled)[0]
        
        # Decode prediction
        prediction_class = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = np.max(probabilities)
        
        # Get class probabilities
        class_probs = dict(zip(class_names, probabilities))
        
        return {
            'prediction_encoded': prediction_encoded,
            'prediction_class': prediction_class,
            'confidence': confidence,
            'probabilities': class_probs,
            'is_anomaly': prediction_class != 0  # Assuming class 0 is normal
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
    model, scaler, label_encoder, feature_names, class_names, model_data = load_model()
    
    if model is None:
        print("❌ Cannot proceed without loaded model")
        return
    
    print(f"\n📋 Feature names: {feature_names}")
    print(f"🏷️  Class names: {class_names}")
    
    # Sample 1: Normal ARP packet (all zeros as baseline)
    normal_sample = np.zeros(len(feature_names))
    print(f"\n🔍 Testing normal sample: {normal_sample[:5]}...")
    
    result1 = predict_class(model, scaler, label_encoder, feature_names, class_names, normal_sample)
    if result1:
        print(f"   Prediction: {result1['prediction_class']} (Class {result1['prediction_encoded']})")
        print(f"   Confidence: {result1['confidence']:.4f}")
        print(f"   Is Anomaly: {result1['is_anomaly']}")
        print(f"   Class Probabilities: {dict(list(result1['probabilities'].items())[:3])}...")
    
    # Sample 2: Slightly modified (some non-zero values)
    modified_sample = np.zeros(len(feature_names))
    modified_sample[0] = 1.0  # Modify first feature
    modified_sample[2] = -0.5  # Modify third feature
    print(f"\n🔍 Testing modified sample: {modified_sample[:5]}...")
    
    result2 = predict_class(model, scaler, label_encoder, feature_names, class_names, modified_sample)
    if result2:
        print(f"   Prediction: {result2['prediction_class']} (Class {result2['prediction_encoded']})")
        print(f"   Confidence: {result2['confidence']:.4f}")
        print(f"   Is Anomaly: {result2['is_anomaly']}")
        print(f"   Class Probabilities: {dict(list(result2['probabilities'].items())[:3])}...")
    
    # Sample 3: Highly anomalous (extreme values)
    extreme_sample = np.zeros(len(feature_names))
    extreme_sample[0] = 10.0  # Very high value
    extreme_sample[1] = -10.0  # Very low value
    print(f"\n🔍 Testing extreme sample: {extreme_sample[:5]}...")
    
    result3 = predict_class(model, scaler, label_encoder, feature_names, class_names, extreme_sample)
    if result3:
        print(f"   Prediction: {result3['prediction_class']} (Class {result3['prediction_encoded']})")
        print(f"   Confidence: {result3['confidence']:.4f}")
        print(f"   Is Anomaly: {result3['is_anomaly']}")
        print(f"   Class Probabilities: {dict(list(result3['probabilities'].items())[:3])}...")

def test_with_real_data_sample():
    """Test with a small sample from the real dataset."""
    print("\n" + "="*50)
    print("TESTING MODEL WITH REAL DATA SAMPLE")
    print("="*50)
    
    # Load model
    model, scaler, label_encoder, feature_names, class_names, model_data = load_model()
    
    if model is None:
        print("❌ Cannot proceed without loaded model")
        return
    
    try:
        # Load a small sample from the dataset
        print("📊 Loading sample from dataset...")
        df_sample = pd.read_csv('../dataset.csv', nrows=10)
        
        # Separate features and labels
        feature_cols = [col for col in df_sample.columns if col != 'label']
        X_sample = df_sample[feature_cols].values
        y_sample = df_sample['label'].values if 'label' in df_sample.columns else None
        
        print(f"📋 Sample shape: {X_sample.shape}")
        print(f"📋 Sample columns: {feature_cols}")
        
        # Test each sample
        for i in range(min(5, len(X_sample))):
            sample = X_sample[i]
            true_label = y_sample[i] if y_sample is not None else "Unknown"
            print(f"\n🔍 Testing real sample {i+1}: {sample[:5]}...")
            print(f"   True label: {true_label}")
            
            result = predict_class(model, scaler, label_encoder, feature_names, class_names, sample)
            if result:
                print(f"   Prediction: {result['prediction_class']} (Class {result['prediction_encoded']})")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Is Anomaly: {result['is_anomaly']}")
                if y_sample is not None:
                    correct = result['prediction_class'] == true_label
                    print(f"   Correct: {'✅' if correct else '❌'}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {str(e)}")

def compare_with_isolation_forest():
    """Compare RandomForest with IsolationForest results."""
    print("\n" + "="*50)
    print("COMPARING RANDOM FOREST WITH ISOLATION FOREST")
    print("="*50)
    
    try:
        # Load RandomForest model
        rf_model, rf_scaler, rf_label_encoder, rf_features, rf_classes, rf_data = load_model('models/rf_model.joblib')
        
        # Load IsolationForest model
        if_model_data = joblib.load('models/iforest.joblib')
        if_model = if_model_data['model']
        if_scaler = if_model_data['scaler']
        if_features = if_model_data['feature_names']
        
        if rf_model is None or if_model is None:
            print("❌ Cannot compare - one or both models not found")
            return
        
        print("✅ Both models loaded successfully!")
        
        # Test with same sample
        test_sample = np.zeros(len(rf_features))
        test_sample[0] = 1.0  # Add some anomaly
        
        print(f"\n🔍 Testing with sample: {test_sample[:5]}...")
        
        # RandomForest prediction
        rf_result = predict_class(rf_model, rf_scaler, rf_label_encoder, rf_features, rf_classes, test_sample)
        if rf_result:
            print(f"RandomForest: {rf_result['prediction_class']} (Confidence: {rf_result['confidence']:.4f})")
        
        # IsolationForest prediction
        if_scaled = if_scaler.transform(test_sample.reshape(1, -1))
        if_prediction = if_model.predict(if_scaled)[0]
        if_score = if_model.decision_function(if_scaled)[0]
        if_anomaly = if_prediction == -1
        
        print(f"IsolationForest: {'ANOMALY' if if_anomaly else 'NORMAL'} (Score: {if_score:.4f})")
        
        print(f"\n📊 Comparison Summary:")
        print(f"   RandomForest: {'Anomaly' if rf_result['is_anomaly'] else 'Normal'}")
        print(f"   IsolationForest: {'Anomaly' if if_anomaly else 'Normal'}")
        print(f"   Agreement: {'✅' if rf_result['is_anomaly'] == if_anomaly else '❌'}")
        
    except Exception as e:
        print(f"❌ Error comparing models: {str(e)}")

def main():
    """Main test function."""
    print("RandomForest ARP Spoofing Detection - Model Testing")
    print("="*60)
    
    # Test model loading and predictions
    test_with_sample_data()
    
    # Test with real data sample
    test_with_real_data_sample()
    
    # Compare with IsolationForest
    compare_with_isolation_forest()
    
    print("\n" + "="*60)
    print("✅ Model testing completed!")
    print("="*60)

if __name__ == "__main__":
    main() 