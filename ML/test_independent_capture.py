#!/usr/bin/env python3
"""
Independent Network Capture Testing
This script tests trained models on independent network captures to validate generalization.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

class IndependentModelTester:
    """Test trained models on independent network captures."""
    
    def __init__(self):
        self.models = {}
        self.test_results = {}
        
    def load_trained_models(self):
        """Load all available trained models."""
        print("Loading trained models...")
        
        model_paths = {
            'full_rf': 'models/rf_model.joblib',
            'robust_rf': 'models/robust_rf_model.joblib',
            'minimal_rf': 'models/minimal_rf_model.joblib',
            'isolation_forest': 'models/iforest.joblib'
        }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    self.models[model_name] = model_data
                    print(f"  ✅ Loaded {model_name}: {model_path}")
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {str(e)}")
            else:
                print(f"  ⚠️  Model not found: {model_path}")
        
        print(f"Loaded {len(self.models)} models")
        return self.models
    
    def prepare_test_data(self, test_filepath):
        """Prepare test data from independent capture."""
        print(f"\nPreparing test data from: {test_filepath}")
        
        if not os.path.exists(test_filepath):
            print(f"❌ Test file not found: {test_filepath}")
            return None
        
        # Load test data
        df_test = pd.read_csv(test_filepath)
        print(f"Test dataset shape: {df_test.shape}")
        print(f"Test columns: {list(df_test.columns)}")
        
        # Check if label column exists
        if 'label' not in df_test.columns:
            print("⚠️  No label column found - will only test unsupervised model")
            return df_test
        
        # Check label distribution
        print(f"Test label distribution:")
        print(df_test['label'].value_counts().sort_index())
        
        return df_test
    
    def test_full_random_forest(self, df_test):
        """Test the full Random Forest model."""
        if 'full_rf' not in self.models:
            print("❌ Full Random Forest model not available")
            return None
        
        print("\n" + "="*50)
        print("TESTING FULL RANDOM FOREST MODEL")
        print("="*50)
        
        model_data = self.models['full_rf']
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        
        # Prepare features
        available_features = [f for f in feature_names if f in df_test.columns]
        missing_features = [f for f in feature_names if f not in df_test.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print(f"Available features: {available_features}")
            return None
        
        # Extract features and labels
        X_test = df_test[available_features].values
        y_test = df_test['label'].values if 'label' in df_test.columns else None
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics if labels available
        results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'feature_names': available_features
        }
        
        if y_test is not None:
            # Encode test labels
            y_test_encoded = label_encoder.transform(y_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='macro')
            precision = precision_score(y_test_encoded, y_pred, average='macro')
            recall = recall_score(y_test_encoded, y_pred, average='macro')
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            class_names = [str(name) for name in label_encoder.classes_]
            print(classification_report(y_test_encoded, y_pred, target_names=class_names))
            
            results.update({
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'y_true': y_test_encoded
            })
        
        return results
    
    def test_robust_random_forest(self, df_test):
        """Test the robust Random Forest model."""
        if 'robust_rf' not in self.models:
            print("❌ Robust Random Forest model not available")
            return None
        
        print("\n" + "="*50)
        print("TESTING ROBUST RANDOM FOREST MODEL")
        print("="*50)
        
        model_data = self.models['robust_rf']
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        
        # Prepare features
        available_features = [f for f in feature_names if f in df_test.columns]
        missing_features = [f for f in feature_names if f not in df_test.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print(f"Available features: {available_features}")
            return None
        
        # Extract features and labels
        X_test = df_test[available_features].values
        y_test = df_test['label'].values if 'label' in df_test.columns else None
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics if labels available
        results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'feature_names': available_features
        }
        
        if y_test is not None:
            # Encode test labels
            y_test_encoded = label_encoder.transform(y_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='macro')
            precision = precision_score(y_test_encoded, y_pred, average='macro')
            recall = recall_score(y_test_encoded, y_pred, average='macro')
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            class_names = [str(name) for name in label_encoder.classes_]
            print(classification_report(y_test_encoded, y_pred, target_names=class_names))
            
            results.update({
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'y_true': y_test_encoded
            })
        
        return results
    
    def test_minimal_random_forest(self, df_test):
        """Test the minimal Random Forest model."""
        if 'minimal_rf' not in self.models:
            print("❌ Minimal Random Forest model not available")
            return None
        
        print("\n" + "="*50)
        print("TESTING MINIMAL RANDOM FOREST MODEL")
        print("="*50)
        
        model_data = self.models['minimal_rf']
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        
        # Prepare features
        available_features = [f for f in feature_names if f in df_test.columns]
        missing_features = [f for f in feature_names if f not in df_test.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print(f"Available features: {available_features}")
            return None
        
        # Extract features and labels
        X_test = df_test[available_features].values
        y_test = df_test['label'].values if 'label' in df_test.columns else None
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics if labels available
        results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'feature_names': available_features
        }
        
        if y_test is not None:
            # Encode test labels
            y_test_encoded = label_encoder.transform(y_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='macro')
            precision = precision_score(y_test_encoded, y_pred, average='macro')
            recall = recall_score(y_test_encoded, y_pred, average='macro')
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            class_names = [str(name) for name in label_encoder.classes_]
            print(classification_report(y_test_encoded, y_pred, target_names=class_names))
            
            results.update({
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'y_true': y_test_encoded
            })
        
        return results
    
    def test_isolation_forest(self, df_test):
        """Test the Isolation Forest model."""
        if 'isolation_forest' not in self.models:
            print("❌ Isolation Forest model not available")
            return None
        
        print("\n" + "="*50)
        print("TESTING ISOLATION FOREST MODEL")
        print("="*50)
        
        model_data = self.models['isolation_forest']
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Prepare features
        available_features = [f for f in feature_names if f in df_test.columns]
        missing_features = [f for f in feature_names if f not in df_test.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print(f"Available features: {available_features}")
            return None
        
        # Extract features
        X_test = df_test[available_features].values
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_scores = model.decision_function(X_test_scaled)
        
        # Calculate anomaly statistics
        n_anomalies = (y_pred == -1).sum()
        n_normal = (y_pred == 1).sum()
        anomaly_rate = n_anomalies / len(y_pred)
        
        print(f"Total samples: {len(y_pred)}")
        print(f"Normal predictions: {n_normal}")
        print(f"Anomaly predictions: {n_anomalies}")
        print(f"Anomaly rate: {anomaly_rate:.4f}")
        print(f"Average anomaly score: {y_scores.mean():.4f}")
        print(f"Score std: {y_scores.std():.4f}")
        
        results = {
            'predictions': y_pred,
            'scores': y_scores,
            'anomaly_rate': anomaly_rate,
            'feature_names': available_features
        }
        
        return results
    
    def compare_model_performance(self):
        """Compare performance across all models."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for model_name, results in self.test_results.items():
            if results and 'accuracy' in results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'F1-Score': results['f1'],
                    'Precision': results['precision'],
                    'Recall': results['recall']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
            
            # Find best model
            best_model = df_comparison.loc[df_comparison['F1-Score'].idxmax()]
            print(f"\n🏆 Best performing model: {best_model['Model']}")
            print(f"   F1-Score: {best_model['F1-Score']:.4f}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        else:
            print("No supervised models with labels available for comparison")
    
    def create_test_visualizations(self, df_test):
        """Create visualizations for test results."""
        print("\nCreating test result visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Independent Test Results', fontsize=16)
        
        # 1. Model comparison (if supervised models available)
        supervised_results = {k: v for k, v in self.test_results.items() 
                            if v and 'accuracy' in v}
        
        if supervised_results:
            model_names = list(supervised_results.keys())
            accuracies = [results['accuracy'] for results in supervised_results.values()]
            f1_scores = [results['f1'] for results in supervised_results.values()]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            axes[0, 0].set_title('Model Performance Comparison')
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Test data distribution
        if 'label' in df_test.columns:
            label_counts = df_test['label'].value_counts().sort_index()
            axes[0, 1].bar(label_counts.index, label_counts.values, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Test Data Label Distribution')
            axes[0, 1].set_xlabel('Label')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Isolation Forest anomaly scores
        if 'isolation_forest' in self.test_results and self.test_results['isolation_forest']:
            scores = self.test_results['isolation_forest']['scores']
            axes[1, 0].hist(scores, bins=50, alpha=0.7, color='orange')
            axes[1, 0].set_title('Isolation Forest Anomaly Scores')
            axes[1, 0].set_xlabel('Anomaly Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance comparison (if available)
        if supervised_results:
            # Show feature importance for best model
            best_model_name = max(supervised_results.keys(), 
                                key=lambda x: supervised_results[x]['f1'])
            if best_model_name in self.models:
                model_data = self.models[best_model_name]
                if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                    importance = model_data['model'].feature_importances_
                    feature_names = model_data['feature_names']
                    
                    # Get top 10 features
                    top_indices = np.argsort(importance)[-10:]
                    top_features = [feature_names[i] for i in top_indices]
                    top_importance = importance[top_indices]
                    
                    axes[1, 1].barh(range(len(top_features)), top_importance)
                    axes[1, 1].set_yticks(range(len(top_features)))
                    axes[1, 1].set_yticklabels(top_features)
                    axes[1, 1].set_title(f'Top 10 Feature Importance ({best_model_name})')
                    axes[1, 1].set_xlabel('Importance')
                    axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/independent_test_results.png', dpi=300, bbox_inches='tight')
        print("Test visualizations saved to: models/independent_test_results.png")
    
    def run_comprehensive_test(self, test_filepath):
        """Run comprehensive testing on independent data."""
        print("ARP Spoofing Detection - Independent Model Testing")
        print("="*60)
        
        # Load models
        self.load_trained_models()
        
        # Prepare test data
        df_test = self.prepare_test_data(test_filepath)
        if df_test is None:
            print("❌ Cannot proceed without test data")
            return
        
        # Test each model
        self.test_results['full_rf'] = self.test_full_random_forest(df_test)
        self.test_results['robust_rf'] = self.test_robust_random_forest(df_test)
        self.test_results['minimal_rf'] = self.test_minimal_random_forest(df_test)
        self.test_results['isolation_forest'] = self.test_isolation_forest(df_test)
        
        # Compare performance
        self.compare_model_performance()
        
        # Create visualizations
        self.create_test_visualizations(df_test)
        
        print(f"\n✅ Independent testing completed!")
        print(f"📊 Check models/independent_test_results.png for visualizations")

def main():
    """Main function for independent testing."""
    # You can specify your own test file here
    test_filepath = '../test_dataset.csv'  # Change this to your test file
    
    tester = IndependentModelTester()
    tester.run_comprehensive_test(test_filepath)

if __name__ == "__main__":
    main() 