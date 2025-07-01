#!/usr/bin/env python3
"""
Quick Model Metrics Visualization
Generate F1 score, accuracy, and other metrics visualizations from existing models
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_model_and_data():
    """Load the best available model and dataset."""
    print("Loading model and data...")
    
    # Try to load the main Random Forest model
    model_path = 'models/rf_model.joblib'
    if not os.path.exists(model_path):
        # Try alternative models
        alternative_models = [
            'models/robust_rf_model.joblib',
            'models/minimal_rf_model.joblib',
            'models/realistic_rf_model.joblib'
        ]
        
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                model_path = alt_model
                break
    
    if not os.path.exists(model_path):
        print("❌ No model found! Please train a model first.")
        return None, None, None
    
    # Load model
    model_data = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Load dataset
    dataset_path = '../dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = '../dataset_clean.csv'
    
    if not os.path.exists(dataset_path):
        print("❌ No dataset found! Using synthetic data for demonstration.")
        return model_data, None, None
    
    # Load real data
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset: {dataset_path}")
    
    # Prepare features and labels
    # Assuming the last column is the label
    feature_columns = df.columns[:-1]  # All columns except the last
    X = df[feature_columns].values
    y = df.iloc[:, -1].values
    
    return model_data, X, y

def generate_metrics_visualizations(model_data, X, y):
    """Generate comprehensive metrics visualizations."""
    print("Generating metrics visualizations...")
    
    # Create output directory
    output_dir = 'metrics_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model components
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [f'Feature_{i}' for i in range(X.shape[1])])
    
    # Split data if we have enough samples
    if len(X) > 100:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_test, y_test = X, y
    
    # Preprocess test data
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate ROC AUC if possible
    roc_auc = None
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        try:
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            pass
    
    print(f"\n📊 Model Performance Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (Macro): {f1_macro:.4f}")
    print(f"   Precision (Macro): {precision_macro:.4f}")
    print(f"   Recall (Macro): {recall_macro:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}" if roc_auc else "   ROC AUC: N/A")
    
    # 1. Create metrics summary plot
    create_metrics_summary_plot(accuracy, f1_macro, precision_macro, recall_macro, roc_auc, output_dir)
    
    # 2. Create confusion matrix
    create_confusion_matrix(y_test, y_pred, output_dir)
    
    # 3. Create ROC curve
    if y_pred_proba is not None:
        create_roc_curve(y_test, y_pred_proba, output_dir)
    
    # 4. Create feature importance plot (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        create_feature_importance_plot(model, feature_names, output_dir)
    
    # 5. Create classification report
    create_classification_report(y_test, y_pred, output_dir)
    
    # 6. Create comprehensive dashboard
    create_comprehensive_dashboard(accuracy, f1_macro, precision_macro, recall_macro, 
                                 roc_auc, y_test, y_pred, y_pred_proba, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    return {
        'accuracy': accuracy,
        'f1_score': f1_macro,
        'precision': precision_macro,
        'recall': recall_macro,
        'roc_auc': roc_auc
    }

def create_metrics_summary_plot(accuracy, f1_score, precision, recall, roc_auc, output_dir):
    """Create a summary plot of key metrics."""
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    values = [accuracy, f1_score, precision, recall]
    
    if roc_auc:
        metrics.append('ROC AUC')
        values.append(roc_auc)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B4513'][:len(metrics)])
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(y_true, y_pred, output_dir):
    """Create confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'ARP Spoof', 'Other'],
                yticklabels=['Normal', 'ARP Spoof', 'Other'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_roc_curve(y_true, y_pred_proba, output_dir):
    """Create ROC curve visualization."""
    plt.figure(figsize=(10, 8))
    
    if y_pred_proba.shape[1] == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    else:
        # Multi-class classification
        for i in range(y_pred_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plot(model, feature_names, output_dir):
    """Create feature importance visualization."""
    feature_importance = model.feature_importances_
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    # Plot top 15 features
    top_features = importance_df.tail(15)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_classification_report(y_true, y_pred, output_dir):
    """Create classification report visualization."""
    report = classification_report(y_true, y_pred, target_names=['Normal', 'ARP Spoof', 'Other'], output_dict=True)
    
    # Create heatmap of precision, recall, f1-score
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop('support', axis=1)  # Remove support column
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(accuracy, f1_score, precision, recall, roc_auc, 
                                 y_true, y_pred, y_pred_proba, output_dir):
    """Create a comprehensive dashboard with all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ARP Spoofing Detection - Model Performance Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Metrics radar chart
    ax1 = axes[0, 0]
    metrics_values = [accuracy, f1_score, precision, recall]
    metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    if roc_auc:
        metrics_values.append(roc_auc)
        metrics_labels.append('ROC AUC')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    metrics_values += metrics_values[:1]  # Close the plot
    angles += angles[:1]
    
    ax1.plot(angles, metrics_values, 'o-', linewidth=2, color='#2E86AB')
    ax1.fill(angles, metrics_values, alpha=0.25, color='#2E86AB')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_labels)
    ax1.set_ylim(0, 1)
    ax1.set_title('Performance Metrics Radar Chart', fontweight='bold')
    ax1.grid(True)
    
    # 2. Prediction confidence distribution
    ax2 = axes[0, 1]
    if y_pred_proba is not None:
        max_proba = np.max(y_pred_proba, axis=1)
        ax2.hist(max_proba, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
        ax2.set_title('Prediction Confidence Distribution', fontweight='bold')
        ax2.set_xlabel('Maximum Probability')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No probability data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Prediction Confidence Distribution', fontweight='bold')
    
    # 3. Confusion matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Normal', 'ARP Spoof', 'Other'],
                yticklabels=['Normal', 'ARP Spoof', 'Other'])
    ax3.set_title('Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Performance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Performance Summary

Accuracy: {accuracy:.4f}
F1-Score: {f1_score:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
ROC AUC: {roc_auc:.4f if roc_auc else 'N/A'}

Dataset Size: {len(y_true)} samples
Classes: {len(np.unique(y_true))}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate metrics visualizations."""
    print("="*60)
    print("QUICK MODEL METRICS VISUALIZATION")
    print("="*60)
    
    # Load model and data
    model_data, X, y = load_model_and_data()
    
    if model_data is None:
        print("❌ Cannot proceed without a model!")
        return
    
    # Generate visualizations
    metrics = generate_metrics_visualizations(model_data, X, y)
    
    print("\n" + "="*60)
    print("✅ METRICS VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"📊 Key Metrics:")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    if metrics['roc_auc']:
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n📁 All visualizations saved to: metrics_output/")
    print(f"   • metrics_summary.png")
    print(f"   • confusion_matrix.png")
    print(f"   • roc_curve.png")
    print(f"   • feature_importance.png")
    print(f"   • classification_report.png")
    print(f"   • comprehensive_dashboard.png")

if __name__ == "__main__":
    main() 