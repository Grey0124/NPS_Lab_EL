#!/usr/bin/env python3
"""
Robust ARP Spoofing Detection Model Training
This script addresses data leakage concerns and implements proper evaluation.
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

class RobustARPSpoofingDetector:
    """Robust RandomForest-based ARP spoofing detector with proper evaluation."""
    
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.suspicious_features = ['frame.number', 'tcp.seq']  # Features to exclude
        
    def load_and_clean_data(self, filepath):
        """Load and clean the dataset, removing suspicious features."""
        print(f"Loading dataset from: {filepath}")
        
        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Original dataset shape: {df.shape}")
        
        # Check for suspicious features
        print(f"\n🔍 Checking for suspicious features: {self.suspicious_features}")
        for feature in self.suspicious_features:
            if feature in df.columns:
                unique_ratio = df[feature].nunique() / len(df)
                print(f"  {feature}: {unique_ratio:.4f} unique ratio")
                if unique_ratio > 0.9:
                    print(f"    ⚠️  Removing {feature} (likely ID/sequence feature)")
        
        # Remove suspicious features
        clean_features = [col for col in df.columns if col != 'label' and col not in self.suspicious_features]
        df_clean = df[clean_features + ['label']]
        
        print(f"Cleaned dataset shape: {df_clean.shape}")
        print(f"Features removed: {[f for f in self.suspicious_features if f in df.columns]}")
        print(f"Remaining features: {len(clean_features)}")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(df_clean['label'].value_counts())
        
        # Separate features and labels
        self.feature_names = clean_features
        X = df_clean[clean_features].values
        y = df_clean['label'].values
        
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
    
    def split_data_properly(self):
        """Split data properly to avoid temporal leakage."""
        print("Splitting data with proper stratification...")
        
        # Use stratified split to maintain class distribution
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
        """Preprocess data with proper scaling."""
        print("Preprocessing data...")
        
        # Fit scaler on training data only (avoid data leakage)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Scaled training data shape: {self.X_train_scaled.shape}")
        print(f"Scaled testing data shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def compute_balanced_weights(self):
        """Compute balanced class weights."""
        print("Computing balanced class weights...")
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        print(f"Class weights: {weight_dict}")
        
        return weight_dict
    
    def train_model(self):
        """Train the RandomForest model with balanced weights."""
        print("Training RandomForestClassifier with balanced weights...")
        
        # Compute balanced weights
        class_weights = self.compute_balanced_weights()
        
        # Initialize RandomForest with conservative parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Reduced to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight=class_weights
        )
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Model training completed!")
        
        return self.model
    
    def evaluate_model_robustly(self):
        """Evaluate the model with multiple metrics and cross-validation."""
        print("Evaluating model performance robustly...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        y_test_proba = self.model.predict_proba(self.X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # F1 scores (macro average for multi-class)
        train_f1 = f1_score(self.y_train, y_train_pred, average='macro')
        test_f1 = f1_score(self.y_test, y_test_pred, average='macro')
        
        # Precision and Recall
        train_precision = precision_score(self.y_train, y_train_pred, average='macro')
        test_precision = precision_score(self.y_test, y_test_pred, average='macro')
        train_recall = recall_score(self.y_train, y_train_pred, average='macro')
        test_recall = recall_score(self.y_test, y_test_pred, average='macro')
        
        # ROC AUC (for multi-class)
        try:
            test_auc = roc_auc_score(self.y_test, y_test_proba, multi_class='ovr', average='macro')
        except:
            test_auc = None
        
        # Cross-validation with stratified k-fold
        print("  Performing stratified cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, 
                                   cv=cv, scoring='f1_macro')
        
        # Print results
        print("\n" + "="*60)
        print("ROBUST MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {test_accuracy:.4f}")
        print(f"Training F1-Score: {train_f1:.4f}")
        print(f"Testing F1-Score:  {test_f1:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Testing Precision:  {test_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")
        print(f"Testing Recall:  {test_recall:.4f}")
        if test_auc:
            print(f"Testing ROC AUC:  {test_auc:.4f}")
        print(f"Cross-validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Check for overfitting
        accuracy_diff = train_accuracy - test_accuracy
        f1_diff = train_f1 - test_f1
        print(f"\nOverfitting Analysis:")
        print(f"  Accuracy difference (train-test): {accuracy_diff:.4f}")
        print(f"  F1 difference (train-test): {f1_diff:.4f}")
        
        if accuracy_diff > 0.05 or f1_diff > 0.05:
            print("  ⚠️  WARNING: Potential overfitting detected!")
        else:
            print("  ✅ No significant overfitting detected")
        
        print("="*60)
        
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
            'test_auc': test_auc,
            'cv_scores': cv_scores,
            'overfitting_accuracy_diff': accuracy_diff,
            'overfitting_f1_diff': f1_diff
        }
        
        return self.metrics
    
    def create_detailed_confusion_matrix(self):
        """Create detailed confusion matrix with class-wise analysis."""
        print("Creating detailed confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        
        # Convert class names to strings
        class_names_str = [str(name) for name in self.class_names]
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Main confusion matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_str,
                   yticklabels=class_names_str)
        plt.title('Confusion Matrix - Robust RandomForest')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Normalized confusion matrix
        plt.subplot(2, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names_str,
                   yticklabels=class_names_str)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Class-wise metrics
        plt.subplot(2, 2, 3)
        class_precision = precision_score(self.y_test, self.y_test_pred, average=None)
        class_recall = recall_score(self.y_test, self.y_test_pred, average=None)
        
        x = np.arange(len(class_names_str))
        width = 0.35
        
        plt.bar(x - width/2, class_precision, width, label='Precision', alpha=0.8)
        plt.bar(x + width/2, class_recall, width, label='Recall', alpha=0.8)
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Class-wise Precision and Recall')
        plt.xticks(x, class_names_str)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance
        plt.subplot(2, 2, 4)
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
        
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'models/robust_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Detailed confusion matrix saved to: {plot_path}")
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.y_test_pred, 
                                  target_names=class_names_str))
        
        return cm
    
    def save_robust_model(self, model_path='models/robust_rf_model.joblib'):
        """Save the robust trained model with metadata."""
        print(f"Saving robust model to: {model_path}")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat(),
            'suspicious_features_removed': self.suspicious_features,
            'model_info': {
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'n_classes': len(self.class_names),
                'test_accuracy': self.metrics['test_accuracy'],
                'test_f1': self.metrics['test_f1'],
                'overfitting_accuracy_diff': self.metrics['overfitting_accuracy_diff'],
                'overfitting_f1_diff': self.metrics['overfitting_f1_diff']
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"Robust model saved successfully!")
        
        return model_path
    
    def print_robust_summary(self):
        """Print a comprehensive summary of the robust training results."""
        print("\n" + "="*60)
        print("ROBUST RANDOM FOREST ARP SPOOFING DETECTOR - SUMMARY")
        print("="*60)
        print(f"Dataset shape: {self.X.shape}")
        print(f"Features used: {len(self.feature_names)}")
        print(f"Features removed: {self.suspicious_features}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Class names: {self.class_names}")
        print(f"Random state: {self.random_state}")
        print(f"Test size: {self.test_size}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Test Accuracy: {self.metrics['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {self.metrics['test_f1']:.4f}")
        print(f"  Test Precision: {self.metrics['test_precision']:.4f}")
        print(f"  Test Recall: {self.metrics['test_recall']:.4f}")
        if self.metrics['test_auc']:
            print(f"  Test ROC AUC: {self.metrics['test_auc']:.4f}")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Accuracy difference: {self.metrics['overfitting_accuracy_diff']:.4f}")
        print(f"  F1 difference: {self.metrics['overfitting_f1_diff']:.4f}")
        
        print(f"\nCross-validation:")
        print(f"  CV F1 mean: {self.metrics['cv_scores'].mean():.4f}")
        print(f"  CV F1 std: {self.metrics['cv_scores'].std():.4f}")
        
        print("="*60)

def main():
    """Main function to train the robust RandomForest model."""
    print("ARP Spoofing Detection - Robust RandomForest Training")
    print("="*60)
    
    # Initialize robust detector
    detector = RobustARPSpoofingDetector(random_state=42, test_size=0.2)
    
    try:
        # Load and clean data
        dataset_path = '../dataset.csv'
        X, y = detector.load_and_clean_data(dataset_path)
        
        # Split data properly
        X_train, X_test, y_train, y_test = detector.split_data_properly()
        
        # Preprocess data
        X_train_scaled, X_test_scaled = detector.preprocess_data()
        
        # Train model
        model = detector.train_model()
        
        # Evaluate model robustly
        metrics = detector.evaluate_model_robustly()
        
        # Create detailed confusion matrix
        cm = detector.create_detailed_confusion_matrix()
        
        # Save robust model
        model_path = detector.save_robust_model()
        
        # Print summary
        detector.print_robust_summary()
        
        print(f"\n✅ Robust training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Results saved to: models/robust_confusion_matrix.png")
        print(f"🧪 Test the model with: python test_robust_model.py")
        
    except Exception as e:
        print(f"❌ Error during robust training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 