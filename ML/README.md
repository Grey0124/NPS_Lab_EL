# ARP Spoofing Detection - Machine Learning Models

This directory contains machine learning models for ARP spoofing detection using both unsupervised (Isolation Forest) and supervised (Random Forest) approaches.

## 📁 Project Structure

```
ML/
├── models/                     # Trained models storage
│   ├── iforest.joblib         # Isolation Forest model
│   └── rf_model.joblib        # Random Forest model (after training)
├── train_isolation_forest.py  # Unsupervised model training
├── train_supervised_model.py  # Supervised model training
├── test_model.py              # Test Isolation Forest model
├── test_supervised_model.py   # Test Random Forest model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── ml_env/                    # Virtual environment
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv ml_env
ml_env\Scripts\activate  # Windows
# OR
source ml_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training

#### Unsupervised Model (Isolation Forest)
```bash
python train_isolation_forest.py
```
- **Purpose**: Detect anomalies without labeled data
- **Output**: `models/iforest.joblib`
- **Features**: Optimized for large datasets, sampling for evaluation

#### Supervised Model (Random Forest)
```bash
python train_supervised_model.py
```
- **Purpose**: Classify ARP packets using labeled data
- **Output**: `models/rf_model.joblib`
- **Features**: Multi-class classification, comprehensive evaluation

### 3. Model Testing

#### Test Isolation Forest
```bash
python test_model.py
```

#### Test Random Forest
```bash
python test_supervised_model.py
```

## 📊 Model Comparison

| Feature | Isolation Forest | Random Forest |
|---------|------------------|---------------|
| **Type** | Unsupervised | Supervised |
| **Input** | Features only | Features + Labels |
| **Output** | Anomaly Score | Class Prediction |
| **Use Case** | Unknown attacks | Known attack types |
| **Training** | Fast | Moderate |
| **Interpretability** | Low | High |

## 🔧 Model Details

### Isolation Forest (Unsupervised)
- **Algorithm**: Isolation Forest for anomaly detection
- **Contamination**: 0.01 (1% expected anomalies)
- **Features**: All numerical features from dataset
- **Output**: Anomaly score (-1 = anomaly, 1 = normal)
- **Optimizations**: 
  - Sampling for large datasets
  - Progress indicators
  - Memory-efficient evaluation

### Random Forest (Supervised)
- **Algorithm**: Random Forest Classifier
- **Classes**: Multi-class classification
- **Features**: All features with label encoding
- **Output**: Class prediction with confidence scores
- **Evaluation Metrics**:
  - Accuracy, F1-Score, Precision, Recall
  - ROC AUC, Cross-validation
  - Confusion Matrix
- **Visualizations**:
  - Feature importance
  - Class distribution
  - Prediction confidence
  - Cross-validation scores

## 📈 Performance Metrics

### Isolation Forest
- Anomaly detection rate
- Silhouette score (sampled)
- Decision function scores

### Random Forest
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC AUC**: Area under ROC curve
- **Cross-validation**: 5-fold CV F1 scores

## 🎯 Usage Examples

### Load and Use Isolation Forest
```python
import joblib

# Load model
model_data = joblib.load('models/iforest.joblib')
model = model_data['model']
scaler = model_data['scaler']

# Predict
sample = [[0, 0, 0, ...]]  # Your features
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
score = model.decision_function(sample_scaled)
```

### Load and Use Random Forest
```python
import joblib

# Load model
model_data = joblib.load('models/rf_model.joblib')
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

# Predict
sample = [[0, 0, 0, ...]]  # Your features
sample_scaled = scaler.transform(sample)
prediction_encoded = model.predict(sample_scaled)[0]
prediction_class = label_encoder.inverse_transform([prediction_encoded])[0]
probabilities = model.predict_proba(sample_scaled)[0]
```

## 🔍 Testing Features

### Isolation Forest Testing
- Sample data testing
- Real dataset validation
- Anomaly score analysis
- Performance comparison

### Random Forest Testing
- Sample data testing
- Real dataset validation
- Class probability analysis
- Model comparison with Isolation Forest
- Prediction confidence assessment

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
pyyaml>=5.4.0
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Error**: 
   - Use smaller dataset samples
   - Reduce model complexity
   - Enable sampling in evaluation

2. **Model Not Found**:
   - Ensure models are trained first
   - Check file paths
   - Verify model directory exists

3. **Dataset Issues**:
   - Check CSV format
   - Verify feature columns
   - Ensure label column exists (for supervised)

### Performance Tips

1. **Large Datasets**:
   - Use sampling for evaluation
   - Enable progress indicators
   - Consider data reduction techniques

2. **Model Optimization**:
   - Tune hyperparameters
   - Use cross-validation
   - Monitor overfitting

## 🔄 Integration with Main Project

These models can be integrated with the main ARP spoofing detection system:

1. **Real-time Detection**: Use trained models in `sniffer.py`
2. **Batch Analysis**: Process captured packets
3. **Alert System**: Trigger alerts based on model predictions
4. **Dashboard**: Visualize detection results

## 📝 Next Steps

1. **Model Integration**: Integrate with main sniffer
2. **Real-time Prediction**: Add live packet analysis
3. **Model Retraining**: Implement periodic retraining
4. **Ensemble Methods**: Combine multiple models
5. **Deep Learning**: Explore neural network approaches

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include test cases
4. Optimize for performance
5. Handle edge cases gracefully 