#!/usr/bin/env python3
"""
Realistic RF Model Visualization
Generate comprehensive visualizations for the realistic_rf_model specifically
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_realistic_rf_model():
    """Load the realistic RF model specifically."""
    print("Loading Realistic RF Model...")
    
    model_path = 'models/realistic_rf_model.joblib'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        model_data = joblib.load(model_path)
        print(f"✅ Successfully loaded: {model_path}")
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_dataset():
    """Load the dataset for evaluation."""
    print("Loading dataset...")
    
    # Try different dataset paths
    dataset_paths = [
        '../dataset_clean.csv',
        '../dataset.csv',
        'dataset_clean.csv',
        'dataset.csv'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"✅ Loaded dataset: {path}")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {len(df.columns)}")
                return df
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue
    
    print("❌ No dataset found! Generating synthetic data for demonstration.")
    return generate_synthetic_data()

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic data for demonstration."""
    print("Generating synthetic data...")
    
    np.random.seed(42)
    
    # Generate realistic ARP features
    features = {
        'frame.time_delta': np.random.exponential(0.1, n_samples),
        'tcp.hdr_len': np.random.choice([20, 24, 28, 32], n_samples),
        'tcp.flag_ack': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'tcp.flag_psh': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'tcp.flag_rst': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'tcp.flag_fin': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'icmp.type': np.random.choice([0, 3, 8, 11], n_samples, p=[0.4, 0.2, 0.3, 0.1]),
        'arp.opcode': np.random.choice([1, 2], n_samples, p=[0.6, 0.4]),
        'arp.hw.size': np.random.choice([6], n_samples),
        'arp.proto.size': np.random.choice([4], n_samples),
        'arp.hw.type': np.random.choice([1], n_samples),
        'arp.proto.type': np.random.choice([2048], n_samples),
        'frame.len': np.random.normal(60, 20, n_samples),
        'frame.cap_len': np.random.normal(60, 20, n_samples),
        'frame.marked': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'frame.ignored': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'frame.time': np.random.uniform(0, 1000, n_samples),
        'frame.time_relative': np.random.uniform(0, 100, n_samples),
        'frame.time_delta_displayed': np.random.exponential(0.1, n_samples),
        'frame.time_epoch': np.random.uniform(1600000000, 1700000000, n_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Generate labels (0: normal, 1: arp_spoof, 2: other_attack)
    y = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
    df['label'] = y
    
    print(f"✅ Generated synthetic data: {df.shape}")
    return df

def prepare_data(df):
    """Prepare features and labels from dataset."""
    print("Preparing data...")
    
    # Assuming the last column is the label
    feature_columns = df.columns[:-1]
    X = df[feature_columns].values
    y = df.iloc[:, -1].values
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Classes: {len(np.unique(y))}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return X, y, feature_columns

def evaluate_model(model_data, X, y, feature_names):
    """Evaluate the realistic RF model."""
    print("Evaluating Realistic RF Model...")
    
    # Extract model components
    model = model_data['model']
    scaler = model_data.get('scaler')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess test data
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    print(f"\n📊 Realistic RF Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (Macro): {f1_macro:.4f}")
    print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"   Precision (Macro): {precision_macro:.4f}")
    print(f"   Recall (Macro): {recall_macro:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'roc_auc': roc_auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_names
    }

def create_visualizations(metrics, model_data):
    """Create comprehensive visualizations."""
    print("Creating visualizations...")
    
    # Create output directory
    output_dir = 'realistic_rf_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance Metrics Summary
    create_performance_summary(metrics, output_dir)
    
    # 2. Confusion Matrix
    create_confusion_matrix(metrics, output_dir)
    
    # 3. ROC Curves
    create_roc_curves(metrics, output_dir)
    
    # 4. Feature Importance
    create_feature_importance(metrics, model_data, output_dir)
    
    # 5. Classification Report
    create_classification_report(metrics, output_dir)
    
    # 6. Comprehensive Dashboard
    create_comprehensive_dashboard(metrics, output_dir)
    
    # 7. Prediction Confidence Analysis
    create_prediction_confidence_analysis(metrics, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")

def create_performance_summary(metrics, output_dir):
    """Create performance metrics summary plot."""
    print("  Creating performance summary...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of key metrics
    key_metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']
    values = [metrics['accuracy'], metrics['f1_macro'], metrics['precision_macro'], 
              metrics['recall_macro'], metrics['roc_auc']]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B4513']
    
    bars = ax1.bar(key_metrics, values, color=colors)
    ax1.set_title('Realistic RF Model - Key Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
    values += values[:1]  # Close the plot
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(key_metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Metrics Radar Chart', fontsize=14, fontweight='bold')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(metrics, output_dir):
    """Create confusion matrix visualization."""
    print("  Creating confusion matrix...")
    
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    unique_classes = np.unique(metrics['y_test'])
    class_names = [f"Class {c}" for c in unique_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Realistic RF Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_roc_curves(metrics, output_dir):
    """Create ROC curves visualization."""
    print("  Creating ROC curves...")
    
    plt.figure(figsize=(12, 8))
    
    # Dynamically get class names
    unique_classes = np.unique(metrics['y_test'])
    class_names = [f"Class {c}" for c in unique_classes]
    
    # Multi-class ROC curves
    for i in range(metrics['y_pred_proba'].shape[1]):
        fpr, tpr, _ = roc_curve(metrics['y_test'] == unique_classes[i], metrics['y_pred_proba'][:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Realistic RF Model - ROC Curves', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance(metrics, model_data, output_dir):
    """Create feature importance visualization."""
    print("  Creating feature importance plot...")
    
    model = model_data['model']
    feature_importance = model.feature_importances_
    feature_names = metrics['feature_names']
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    # Plot top 15 features
    top_features = importance_df.tail(15)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Realistic RF Model - Top 15 Feature Importance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_classification_report(metrics, output_dir):
    """Create classification report visualization."""
    print("  Creating classification report...")
    
    unique_classes = np.unique(metrics['y_test'])
    class_names = [f"Class {c}" for c in unique_classes]
    report = classification_report(metrics['y_test'], metrics['y_pred'], 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Create heatmap of precision, recall, f1-score
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop('support', axis=1, errors='ignore')  # Remove support column if present
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Realistic RF Model - Classification Report', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(metrics, output_dir):
    """Create comprehensive dashboard."""
    print("  Creating comprehensive dashboard...")
    
    unique_classes = np.unique(metrics['y_test'])
    class_names = [f"Class {c}" for c in unique_classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Realistic RF Model - Comprehensive Performance Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Metrics radar chart
    ax1 = axes[0, 0]
    metrics_values = [metrics['accuracy'], metrics['f1_macro'], metrics['precision_macro'], 
                     metrics['recall_macro'], metrics['roc_auc']]
    metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']
    
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
    max_proba = np.max(metrics['y_pred_proba'], axis=1)
    ax2.hist(max_proba, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
    ax2.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax2.set_xlabel('Maximum Probability')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=class_names,
                yticklabels=class_names)
    ax3.set_title('Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Performance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Realistic RF Model Performance Summary

Accuracy: {metrics['accuracy']:.4f}
F1-Score (Macro): {metrics['f1_macro']:.4f}
F1-Score (Weighted): {metrics['f1_weighted']:.4f}
Precision (Macro): {metrics['precision_macro']:.4f}
Recall (Macro): {metrics['recall_macro']:.4f}
ROC AUC: {metrics['roc_auc']:.4f}

Test Set Size: {len(metrics['y_test'])} samples
Classes: {len(np.unique(metrics['y_test']))}
Model Type: Random Forest (Realistic)
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_confidence_analysis(metrics, output_dir):
    """Create prediction confidence analysis."""
    print("  Creating prediction confidence analysis...")
    
    unique_classes = np.unique(metrics['y_test'])
    class_names = [f"Class {c}" for c in unique_classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Realistic RF Model - Prediction Confidence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confidence distribution by class
    ax1 = axes[0, 0]
    for i in range(metrics['y_pred_proba'].shape[1]):
        class_confidences = metrics['y_pred_proba'][metrics['y_pred'] == unique_classes[i], i]
        ax1.hist(class_confidences, bins=20, alpha=0.6, label=class_names[i])
    
    ax1.set_title('Confidence Distribution by Predicted Class')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence vs accuracy
    ax2 = axes[0, 1]
    correct_predictions = metrics['y_test'] == metrics['y_pred']
    max_proba = np.max(metrics['y_pred_proba'], axis=1)
    
    ax2.scatter(max_proba[correct_predictions], np.ones(np.sum(correct_predictions)), 
               alpha=0.6, color='green', label='Correct', s=20)
    ax2.scatter(max_proba[~correct_predictions], np.zeros(np.sum(~correct_predictions)), 
               alpha=0.6, color='red', label='Incorrect', s=20)
    ax2.set_title('Prediction Confidence vs Accuracy')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Correct (1) / Incorrect (0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Class-wise confidence boxplot
    ax3 = axes[1, 0]
    confidence_by_class = []
    for i in range(metrics['y_pred_proba'].shape[1]):
        class_confidences = metrics['y_pred_proba'][metrics['y_pred'] == unique_classes[i], i]
        confidence_by_class.append(class_confidences)
    
    ax3.boxplot(confidence_by_class, labels=class_names)
    ax3.set_title('Confidence Distribution by Class (Boxplot)')
    ax3.set_ylabel('Prediction Confidence')
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy at different confidence thresholds
    ax4 = axes[1, 1]
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracies = []
    
    for threshold in thresholds:
        high_conf_mask = max_proba >= threshold
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(correct_predictions[high_conf_mask])
            accuracies.append(high_conf_accuracy)
        else:
            accuracies.append(0)
    
    ax4.plot(thresholds, accuracies, 'o-', linewidth=2, color='#2E86AB')
    ax4.set_title('Accuracy at Different Confidence Thresholds')
    ax4.set_xlabel('Confidence Threshold')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate realistic RF model visualizations."""
    print("="*70)
    print("REALISTIC RF MODEL VISUALIZATION GENERATOR")
    print("="*70)
    
    # Load model
    model_data = load_realistic_rf_model()
    if model_data is None:
        print("❌ Cannot proceed without the realistic RF model!")
        return
    
    # Load dataset
    df = load_dataset()
    if df is None:
        print("❌ Cannot proceed without data!")
        return
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Evaluate model
    metrics = evaluate_model(model_data, X, y, feature_names)
    
    # Create visualizations
    create_visualizations(metrics, model_data)
    
    print("\n" + "="*70)
    print("✅ REALISTIC RF MODEL VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"📊 Key Metrics for Realistic RF Model:")
    print(f"   F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"   Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n📁 All visualizations saved to: realistic_rf_visualizations/")
    print(f"   • performance_summary.png")
    print(f"   • confusion_matrix.png")
    print(f"   • roc_curves.png")
    print(f"   • feature_importance.png")
    print(f"   • classification_report.png")
    print(f"   • comprehensive_dashboard.png")
    print(f"   • prediction_confidence_analysis.png")

if __name__ == "__main__":
    main() 