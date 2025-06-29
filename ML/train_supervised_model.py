#!/usr/bin/env python3
"""
Supervised Model Training for ARP Spoofing Detection

This script loads the ARP spoofing dataset, trains a RandomForestClassifier,
evaluates its performance with F1-score, accuracy, and confusion matrix,
and saves the trained model for deployment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SupervisedARPSpoofingDetector:
    """RandomForest-based supervised ARP spoofing detector."""
    
    def __init__(self, random_state=42, test_size=0.2):
        """
        Initialize the supervised ARP spoofing detector.
        
        Args:
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, filepath):
        """
        Load and preprocess the ARP spoofing dataset.
        
        Args:
            filepath (str): Path to the dataset CSV file
        """
        print(f"Loading dataset from: {filepath}")
        
        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for label column
        if 'label' in df.columns:
            print(f"Label distribution:\n{df['label'].value_counts()}")
            
            # Separate features and labels
            self.feature_names = [col for col in df.columns if col != 'label']
            X = df[self.feature_names].values
            y = df['label'].values
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
            
            print(f"Feature columns: {self.feature_names}")
            print(f"Number of features: {len(self.feature_names)}")
            print(f"Number of classes: {len(self.class_names)}")
            print(f"Class mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            
        else:
            raise ValueError("No 'label' column found in dataset. Supervised learning requires labels.")
        
        # Store data
        self.X = X
        self.y = y
        self.y_encoded = y_encoded
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Label vector shape: {self.y_encoded.shape}")
        
        return self.X, self.y_encoded
    
    def split_data(self):
        """Split data into training and testing sets."""
        print("Splitting data into training and testing sets...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y_encoded  # Ensure balanced split
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
        """Preprocess the data using standardization."""
        print("Preprocessing data...")
        
        # Fit scaler on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Scaled training data shape: {self.X_train_scaled.shape}")
        print(f"Scaled testing data shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_model(self):
        """Train the RandomForest model."""
        print("Training RandomForestClassifier...")
        
        # Initialize RandomForest with optimized parameters for ARP spoofing detection
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Model training completed!")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance comprehensively."""
        print("Evaluating model performance...")
        
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
        
        # Cross-validation score
        print("  Performing cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5, scoring='f1_macro')
        
        # Print results
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
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
            'cv_scores': cv_scores
        }
        
        return self.metrics
    
    def create_confusion_matrix(self):
        """Create and display confusion matrix."""
        print("Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        
        # Convert class names to strings for compatibility
        class_names_str = [str(name) for name in self.class_names]
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_str,
                   yticklabels=class_names_str)
        plt.title('Confusion Matrix - RandomForest ARP Spoofing Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = 'models/confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {plot_path}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_test_pred, 
                                  target_names=class_names_str))
        
        return cm
    
    def visualize_results(self, save_plots=True):
        """Visualize the model results."""
        print("Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RandomForest ARP Spoofing Detection Results', fontsize=16)
        
        # 1. Feature Importance
        print("  Creating feature importance plot...")
        feature_importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot top 10 features
        top_features = feature_importance_df.tail(10)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_title('Top 10 Feature Importance')
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class Distribution
        print("  Creating class distribution plot...")
        unique, counts = np.unique(self.y, return_counts=True)
        axes[0, 1].bar(unique, counts, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Class Distribution in Dataset')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(unique)
        # Convert class names to strings for display
        class_names_str = [str(name) for name in self.class_names]
        axes[0, 1].set_xticklabels(class_names_str)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction vs Actual (scatter plot of probabilities)
        print("  Creating prediction probability plot...")
        max_proba = np.max(self.y_test_proba, axis=1)
        axes[1, 0].scatter(range(len(max_proba)), max_proba, 
                          c=self.y_test, cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Prediction Confidence vs True Labels')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Maximum Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cross-validation scores
        print("  Creating cross-validation plot...")
        cv_scores = self.metrics['cv_scores']
        axes[1, 1].bar(range(1, len(cv_scores) + 1), cv_scores, color='orange', alpha=0.7)
        axes[1, 1].axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                          label=f'Mean: {cv_scores.mean():.4f}')
        axes[1, 1].set_title('Cross-validation F1 Scores')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = 'models/randomforest_results.png'
            print(f"  Saving plots to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
        
        plt.show()
    
    def save_model(self, model_path='models/rf_model.joblib'):
        """Save the trained model and components."""
        print(f"Saving model to: {model_path}")
        
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
            'model_info': {
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'n_classes': len(self.class_names),
                'test_accuracy': self.metrics['test_accuracy'],
                'test_f1': self.metrics['test_f1']
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved successfully!")
        
        return model_path
    
    def print_summary(self):
        """Print a summary of the training results."""
        print("\n" + "="*60)
        print("RANDOM FOREST ARP SPOOFING DETECTOR - TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)}")
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
        
        print("="*60)

def main():
    """Main function to train the RandomForest model."""
    print("ARP Spoofing Detection - RandomForest Supervised Training")
    print("="*60)
    
    # Initialize detector
    detector = SupervisedARPSpoofingDetector(random_state=42, test_size=0.2)
    
    try:
        # Load data
        dataset_path = '../dataset.csv'  # Relative to ML folder
        X, y = detector.load_data(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = detector.split_data()
        
        # Preprocess data
        X_train_scaled, X_test_scaled = detector.preprocess_data()
        
        # Train model
        model = detector.train_model()
        
        # Evaluate model
        metrics = detector.evaluate_model()
        
        # Create confusion matrix
        cm = detector.create_confusion_matrix()
        
        # Visualize results
        detector.visualize_results(save_plots=True)
        
        # Save model
        model_path = detector.save_model()
        
        # Print summary
        detector.print_summary()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Results saved to: models/randomforest_results.png")
        print(f"📈 Confusion matrix saved to: models/confusion_matrix.png")
        print(f"🧪 Test the model with: python test_supervised_model.py")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 