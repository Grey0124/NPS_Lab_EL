#!/usr/bin/env python3
"""
Realistic ARP Spoofing Detection Model Training
This script removes all data leakage sources and trains on genuine network behavior.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealisticARPSpoofingDetector:
    """Realistic RandomForest-based ARP spoofing detector with no data leakage."""
    
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
        # Only use genuine network behavior features
        # Exclude all features with data leakage
        self.realistic_features = [
            'frame.time_delta',      # Time between packets (genuine network timing)
            'tcp.hdr_len',           # TCP header length (network protocol behavior)
            'tcp.flag_ack',          # TCP ACK flag (protocol state)
            'tcp.flag_psh',          # TCP PSH flag (protocol state)
            'tcp.flag_rst',          # TCP RST flag (protocol state)
            'tcp.flag_fin',          # TCP FIN flag (protocol state)
            'icmp.type'              # ICMP type (protocol behavior)
        ]
        
        # Features to exclude due to data leakage
        self.excluded_features = [
            'frame.number',          # Perfect mapping to labels
            'data.len',              # High correlation (0.7210)
            'tcp.flag_syn',          # High correlation (0.5203)
            'tcp.seq',               # High deterministic relationship
            'arp.opcode',            # Deterministic mapping per class
            'arp.src.hw_mac',        # Deterministic mapping per class
            'arp.dst.hw_mac'         # Deterministic mapping per class
        ]
        
    def load_and_clean_data(self, filepath):
        """Load dataset and remove all data leakage sources."""
        print(f"Loading dataset from: {filepath}")
        
        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Original dataset shape: {df.shape}")
        
        # Check for realistic features
        missing_features = [f for f in self.realistic_features if f not in df.columns]
        if missing_features:
            print(f"⚠️  Missing realistic features: {missing_features}")
            # Use only available features
            self.realistic_features = [f for f in self.realistic_features if f in df.columns]
        
        # Select only realistic features and label
        df_realistic = df[self.realistic_features + ['label']]
        
        print(f"Realistic dataset shape: {df_realistic.shape}")
        print(f"Features used: {self.realistic_features}")
        print(f"Features excluded: {self.excluded_features}")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(df_realistic['label'].value_counts())
        
        # Analyze feature distributions
        print(f"\nRealistic feature analysis:")
        for feature in self.realistic_features:
            unique_vals = df_realistic[feature].nunique()
            print(f"  {feature}: {unique_vals} unique values")
            
            # Check correlation with label
            if df_realistic[feature].dtype in ['int64', 'float64']:
                corr = abs(df_realistic[feature].corr(df_realistic['label']))
                print(f"    Correlation with label: {corr:.4f}")
                if corr > 0.3:
                    print(f"    ⚠️  WARNING: Moderate correlation detected!")
        
        # Separate features and labels
        self.feature_names = self.realistic_features
        X = df_realistic[self.realistic_features].values
        y = df_realistic['label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label vector shape: {y_encoded.shape}")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Store data
        self.X = X
        self.y = y
        self.y_encoded = y_encoded
        
        return self.X, self.y_encoded
    
    def split_data(self):
        """Split data with stratification."""
        print("Splitting data with stratification...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y_encoded
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Training labels distribution: {np.bincount(y_train)}")
        print(f"Testing labels distribution: {np.bincount(y_test)}")
        
        # Store splits
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self):
        """Preprocess data."""
        print("Preprocessing data...")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Scaled training data shape: {self.X_train_scaled.shape}")
        print(f"Scaled testing data shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_model(self):
        """Train the realistic RandomForest model."""
        print("Training realistic RandomForestClassifier...")
        
        # Compute balanced weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        print(f"Class weights: {weight_dict}")
        
        # Initialize RandomForest with realistic parameters
        self.model = RandomForestClassifier(
            n_estimators=200,        # More trees for complex patterns
            max_depth=10,            # Allow deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight=weight_dict
        )
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Realistic model training completed!")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the realistic model."""
        print("Evaluating realistic model performance...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        y_test_proba = self.model.predict_proba(self.X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        train_f1 = f1_score(self.y_train, y_train_pred, average='macro')
        test_f1 = f1_score(self.y_test, y_test_pred, average='macro')
        
        train_precision = precision_score(self.y_train, y_train_pred, average='macro')
        test_precision = precision_score(self.y_test, y_test_pred, average='macro')
        train_recall = recall_score(self.y_train, y_train_pred, average='macro')
        test_recall = recall_score(self.y_test, y_test_pred, average='macro')
        
        # Cross-validation
        print("  Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, 
                                   cv=cv, scoring='f1_macro')
        
        # Print results
        print("\n" + "="*60)
        print("REALISTIC MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {test_accuracy:.4f}")
        print(f"Training F1-Score: {train_f1:.4f}")
        print(f"Testing F1-Score:  {test_f1:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Testing Precision:  {test_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")
        print(f"Testing Recall:  {test_recall:.4f}")
        print(f"Cross-validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Overfitting analysis
        accuracy_diff = train_accuracy - test_accuracy
        f1_diff = train_f1 - test_f1
        print(f"\nOverfitting Analysis:")
        print(f"  Accuracy difference (train-test): {accuracy_diff:.4f}")
        print(f"  F1 difference (train-test): {f1_diff:.4f}")
        
        if accuracy_diff > 0.05 or f1_diff > 0.05:
            print("  ⚠️  WARNING: Potential overfitting detected!")
        else:
            print("  ✅ No significant overfitting detected")
        
        # Store results
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        self.y_test_proba = y_test_proba
        self.metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'cv_scores': cv_scores,
            'overfitting_accuracy_diff': accuracy_diff,
            'overfitting_f1_diff': f1_diff
        }
        
        return self.metrics
    
    def create_realistic_confusion_matrix(self):
        """Create confusion matrix for realistic model."""
        print("Creating realistic model confusion matrix...")
        
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        class_names_str = [str(name) for name in self.class_names]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_str,
                   yticklabels=class_names_str)
        plt.title('Confusion Matrix - Realistic Model (No Data Leakage)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plot_path = 'models/realistic_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Realistic confusion matrix saved to: {plot_path}")
        
        print("\nRealistic Model Classification Report:")
        print(classification_report(self.y_test, self.y_test_pred, 
                                  target_names=class_names_str))
        
        return cm
    
    def save_realistic_model(self, model_path='models/realistic_rf_model.joblib'):
        """Save the realistic trained model."""
        print(f"Saving realistic model to: {model_path}")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat(),
            'excluded_features': self.excluded_features,
            'model_info': {
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'n_classes': len(self.class_names),
                'test_accuracy': self.metrics['test_accuracy'],
                'test_f1': self.metrics['test_f1'],
                'features_used': self.realistic_features
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"Realistic model saved successfully!")
        
        return model_path
    
    def print_realistic_summary(self):
        """Print summary of realistic model results."""
        print("\n" + "="*60)
        print("REALISTIC ARP SPOOFING DETECTOR - SUMMARY")
        print("="*60)
        print(f"Features used: {self.realistic_features}")
        print(f"Features excluded: {self.excluded_features}")
        print(f"Dataset shape: {self.X.shape}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Class names: {self.class_names}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Test Accuracy: {self.metrics['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {self.metrics['test_f1']:.4f}")
        print(f"  Test Precision: {self.metrics['test_precision']:.4f}")
        print(f"  Test Recall: {self.metrics['test_recall']:.4f}")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Accuracy difference: {self.metrics['overfitting_accuracy_diff']:.4f}")
        print(f"  F1 difference: {self.metrics['overfitting_f1_diff']:.4f}")
        
        print(f"\nCross-validation:")
        print(f"  CV F1 mean: {self.metrics['cv_scores'].mean():.4f}")
        print(f"  CV F1 std: {self.metrics['cv_scores'].std():.4f}")
        
        print("="*60)

def main():
    """Main function to train the realistic model."""
    print("ARP Spoofing Detection - Realistic Model Training")
    print("="*60)
    
    detector = RealisticARPSpoofingDetector(random_state=42, test_size=0.2)
    
    try:
        # Load and clean data
        dataset_path = '../dataset.csv'
        X, y = detector.load_and_clean_data(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = detector.split_data()
        
        # Preprocess data
        X_train_scaled, X_test_scaled = detector.preprocess_data()
        
        # Train model
        model = detector.train_model()
        
        # Evaluate model
        metrics = detector.evaluate_model()
        
        # Create confusion matrix
        cm = detector.create_realistic_confusion_matrix()
        
        # Save model
        model_path = detector.save_realistic_model()
        
        # Print summary
        detector.print_realistic_summary()
        
        print(f"\n✅ Realistic model training completed!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Results saved to: models/realistic_confusion_matrix.png")
        
    except Exception as e:
        print(f"❌ Error during realistic training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 