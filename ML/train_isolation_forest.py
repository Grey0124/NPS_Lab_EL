#!/usr/bin/env python3
"""
IsolationForest Training for ARP Spoofing Detection

This script loads the ARP spoofing dataset, trains an IsolationForest model,
evaluates its performance, and saves the trained model for deployment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ARPSpoofingDetector:
    """IsolationForest-based ARP spoofing detector."""
    
    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize the ARP spoofing detector.
        
        Args:
            contamination (float): Expected proportion of anomalies (default: 0.01)
            random_state (int): Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
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
            # Store labels for evaluation (but don't use for training)
            self.labels = df['label'].values
            # Remove label column for unsupervised training
            df = df.drop('label', axis=1)
        else:
            print("No label column found - using unsupervised approach only")
            self.labels = None
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        print(f"Feature columns: {self.feature_names}")
        
        # Convert to numpy array
        self.X = df.values
        print(f"Feature matrix shape: {self.X.shape}")
        
        return self.X
    
    def preprocess_data(self):
        """Preprocess the data using standardization."""
        print("Preprocessing data...")
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"Scaled data shape: {self.X_scaled.shape}")
        
        return self.X_scaled
    
    def train_model(self):
        """Train the IsolationForest model."""
        print(f"Training IsolationForest with contamination={self.contamination}")
        
        # Initialize IsolationForest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            bootstrap=True
        )
        
        # Train the model
        self.model.fit(self.X_scaled)
        print("Model training completed!")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        print("Evaluating model performance...")
        
        # Get anomaly scores
        print("  Calculating anomaly scores...")
        anomaly_scores = self.model.decision_function(self.X_scaled)
        predictions = self.model.predict(self.X_scaled)
        
        # Convert predictions: -1 for anomalies, 1 for normal
        anomaly_predictions = (predictions == -1).astype(int)
        
        print(f"Anomaly scores range: {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")
        print(f"Predicted anomalies: {anomaly_predictions.sum()} out of {len(anomaly_predictions)}")
        print(f"Anomaly rate: {anomaly_predictions.sum() / len(anomaly_predictions):.4f}")
        
        # Calculate silhouette score if labels are available and dataset is not too large
        if self.labels is not None:
            try:
                # For large datasets, use a sample for silhouette score
                if len(self.X_scaled) > 10000:
                    print("  Dataset is large, using sample for silhouette score calculation...")
                    # Sample 10,000 points for silhouette calculation
                    sample_indices = np.random.choice(len(self.X_scaled), 10000, replace=False)
                    sample_data = self.X_scaled[sample_indices]
                    sample_labels = self.labels[sample_indices]
                    silhouette_avg = silhouette_score(sample_data, sample_labels)
                    print(f"Silhouette Score (sampled): {silhouette_avg:.4f}")
                else:
                    print("  Calculating silhouette score...")
                    silhouette_avg = silhouette_score(self.X_scaled, self.labels)
                    print(f"Silhouette Score: {silhouette_avg:.4f}")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
        
        # Store evaluation results
        self.anomaly_scores = anomaly_scores
        self.predictions = predictions
        self.anomaly_predictions = anomaly_predictions
        
        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'anomaly_predictions': anomaly_predictions
        }
    
    def visualize_results(self, save_plots=True):
        """Visualize the model results."""
        print("Creating visualizations...")
        
        # For large datasets, sample data for visualization
        if len(self.X_scaled) > 50000:
            print("  Dataset is large, sampling for visualization...")
            sample_size = 50000
            sample_indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[sample_indices]
            scores_sample = self.anomaly_scores[sample_indices]
            predictions_sample = self.predictions[sample_indices]
        else:
            X_sample = self.X_scaled
            scores_sample = self.anomaly_scores
            predictions_sample = self.predictions
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IsolationForest ARP Spoofing Detection Results', fontsize=16)
        
        print("  Creating anomaly score distribution...")
        # 1. Anomaly Score Distribution
        axes[0, 0].hist(scores_sample, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Anomaly Score Distribution')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        print("  Creating scatter plot...")
        # 2. Anomaly vs Normal Points
        normal_mask = predictions_sample == 1
        anomaly_mask = predictions_sample == -1
        
        # Use first two features for visualization
        feature1, feature2 = 0, 1
        axes[0, 1].scatter(X_sample[normal_mask, feature1], 
                          X_sample[normal_mask, feature2], 
                          c='blue', alpha=0.6, label='Normal', s=20)
        axes[0, 1].scatter(X_sample[anomaly_mask, feature1], 
                          X_sample[anomaly_mask, feature2], 
                          c='red', alpha=0.8, label='Anomaly', s=30)
        axes[0, 1].set_title(f'Anomaly Detection Results\n(Features: {self.feature_names[feature1]}, {self.feature_names[feature2]})')
        axes[0, 1].set_xlabel(self.feature_names[feature1])
        axes[0, 1].set_ylabel(self.feature_names[feature2])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        print("  Creating score timeline...")
        # 3. Anomaly Score vs Sample Index (use sample for large datasets)
        if len(scores_sample) > 10000:
            # Sample for timeline plot
            timeline_indices = np.linspace(0, len(scores_sample)-1, 10000, dtype=int)
            timeline_scores = scores_sample[timeline_indices]
            axes[1, 0].plot(timeline_scores, alpha=0.7, color='green')
        else:
            axes[1, 0].plot(scores_sample, alpha=0.7, color='green')
        
        axes[1, 0].axhline(y=self.model.threshold_, color='red', linestyle='--', 
                          label=f'Threshold: {self.model.threshold_:.4f}')
        axes[1, 0].set_title('Anomaly Scores Over Time')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Anomaly Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        print("  Calculating feature importance...")
        # 4. Feature Importance (based on anomaly score correlation)
        if len(self.feature_names) > 1:
            correlations = []
            for i, feature in enumerate(self.feature_names):
                corr = np.corrcoef(X_sample[:, i], scores_sample)[0, 1]
                correlations.append(abs(corr))
            
            # Sort features by correlation
            feature_importance = list(zip(self.feature_names, correlations))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*feature_importance[:10])  # Top 10 features
            
            axes[1, 1].barh(range(len(features)), importances, color='orange', alpha=0.7)
            axes[1, 1].set_yticks(range(len(features)))
            axes[1, 1].set_yticklabels(features)
            axes[1, 1].set_title('Feature Importance (Correlation with Anomaly Score)')
            axes[1, 1].set_xlabel('Absolute Correlation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = 'models/isolation_forest_results.png'
            print(f"  Saving plots to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
        
        plt.show()
    
    def save_model(self, model_path='models/iforest.joblib'):
        """Save the trained model and scaler."""
        print(f"Saving model to: {model_path}")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat(),
            'model_info': {
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'anomaly_rate': self.anomaly_predictions.sum() / len(self.anomaly_predictions) if hasattr(self, 'anomaly_predictions') else None
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved successfully!")
        
        return model_path
    
    def print_summary(self):
        """Print a summary of the training results."""
        print("\n" + "="*60)
        print("ISOLATION FOREST ARP SPOOFING DETECTOR - TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Contamination: {self.contamination}")
        print(f"Random state: {self.random_state}")
        
        if hasattr(self, 'anomaly_predictions'):
            print(f"Predicted anomalies: {self.anomaly_predictions.sum()}")
            print(f"Anomaly rate: {self.anomaly_predictions.sum() / len(self.anomaly_predictions):.4f}")
        
        if self.labels is not None:
            print(f"True labels available: Yes")
            print(f"Label distribution: {np.bincount(self.labels)}")
        
        print("="*60)

def main():
    """Main function to train the IsolationForest model."""
    print("ARP Spoofing Detection - IsolationForest Training")
    print("="*60)
    
    # Initialize detector
    detector = ARPSpoofingDetector(contamination=0.01, random_state=42)
    
    try:
        # Load data
        dataset_path = '../dataset.csv'  # Relative to ML folder
        X = detector.load_data(dataset_path)
        
        # Preprocess data
        X_scaled = detector.preprocess_data()
        
        # Train model
        model = detector.train_model()
        
        # Evaluate model
        results = detector.evaluate_model()
        
        # Visualize results (skip for very large datasets to prevent hanging)
        if len(X_scaled) > 100000:
            print("\n⚠️  Dataset is very large (>100k samples). Skipping visualization to prevent hanging.")
            print("   You can still test the model using: python test_model.py")
            save_plots = False
        else:
            save_plots = True
            
        if save_plots:
            detector.visualize_results(save_plots=True)
        
        # Save model
        model_path = detector.save_model()
        
        # Print summary
        detector.print_summary()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        if save_plots:
            print(f"📊 Results saved to: models/isolation_forest_results.png")
        print(f"🧪 Test the model with: python test_model.py")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 