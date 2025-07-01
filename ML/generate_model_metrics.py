#!/usr/bin/env python3
"""
Comprehensive Model Metrics Visualization Generator
Generates visual outputs for all ARP spoofing detection model metrics
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelMetricsVisualizer:
    """Comprehensive model metrics visualization generator."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.output_dir = 'metrics_visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Available models
        self.model_files = {
            'Random Forest': 'rf_model.joblib',
            'Robust RF': 'robust_rf_model.joblib',
            'Minimal RF': 'minimal_rf_model.joblib',
            'Realistic RF': 'realistic_rf_model.joblib',
            'Isolation Forest': 'iforest.joblib'
        }
        
        self.models = {}
        self.metrics = {}
        
    def load_models(self):
        """Load all available models."""
        print("Loading models...")
        
        for model_name, filename in self.model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                try:
                    model_data = joblib.load(filepath)
                    self.models[model_name] = model_data
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            else:
                print(f"⚠️  Model file not found: {filepath}")
    
    def generate_test_data(self, n_samples=1000):
        """Generate synthetic test data for evaluation."""
        print("Generating test data...")
        
        # Create synthetic features similar to real ARP data
        np.random.seed(42)
        
        # Generate realistic feature values
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
        
        X = pd.DataFrame(features)
        
        # Generate labels (0: normal, 1: arp_spoof, 2: other_attack)
        y = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
        
        return X, y
    
    def evaluate_model(self, model_name, model_data, X_test, y_test):
        """Evaluate a single model and return metrics."""
        print(f"Evaluating {model_name}...")
        
        try:
            model = model_data['model']
            scaler = model_data.get('scaler')
            label_encoder = model_data.get('label_encoder')
            
            # Preprocess test data
            if scaler:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
                y_pred = model.predict(X_test_scaled)
            else:
                # For Isolation Forest
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'true_labels': y_test
            }
            
            # Calculate ROC AUC if possible
            if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                try:
                    if y_pred_proba.shape[1] == 2:
                        # Binary classification
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        # Multi-class classification
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    metrics['roc_auc'] = None
            else:
                metrics['roc_auc'] = None
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def create_comprehensive_metrics_dashboard(self):
        """Create a comprehensive metrics dashboard."""
        print("Creating comprehensive metrics dashboard...")
        
        # Generate test data
        X_test, y_test = self.generate_test_data(2000)
        
        # Evaluate all models
        for model_name, model_data in self.models.items():
            metrics = self.evaluate_model(model_name, model_data, X_test, y_test)
            if metrics:
                self.metrics[model_name] = metrics
        
        # Create comprehensive visualization
        self.create_metrics_comparison_plot()
        self.create_individual_model_plots()
        self.create_confusion_matrices()
        self.create_roc_curves()
        self.create_precision_recall_curves()
        self.create_feature_importance_plots()
        self.create_performance_summary_table()
    
    def create_metrics_comparison_plot(self):
        """Create comparison plot of all models."""
        print("Creating metrics comparison plot...")
        
        # Prepare data for plotting
        metrics_data = []
        for model_name, metrics in self.metrics.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'F1-Score (Weighted)': metrics['f1_weighted'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'ROC AUC': metrics['roc_auc'] if metrics['roc_auc'] else 0
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)', 
                          'Precision (Macro)', 'Recall (Macro)', 'ROC AUC']
        
        for i, metric in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            bars = ax.bar(df['Model'], df[metric], color=sns.color_palette("husl", len(df)))
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison saved to: {self.output_dir}/model_comparison.png")
        plt.show()
    
    def create_individual_model_plots(self):
        """Create individual detailed plots for each model."""
        print("Creating individual model plots...")
        
        for model_name, metrics in self.metrics.items():
            print(f"  Creating plots for {model_name}...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{model_name} - Detailed Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Metrics radar chart
            ax1 = axes[0, 0]
            metrics_values = [
                metrics['accuracy'],
                metrics['f1_macro'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['roc_auc'] if metrics['roc_auc'] else 0
            ]
            metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
            metrics_values += metrics_values[:1]  # Close the plot
            angles += angles[:1]
            
            ax1.plot(angles, metrics_values, 'o-', linewidth=2, label=model_name)
            ax1.fill(angles, metrics_values, alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(metrics_labels)
            ax1.set_ylim(0, 1)
            ax1.set_title('Performance Metrics Radar Chart')
            ax1.grid(True)
            
            # 2. Prediction confidence distribution
            ax2 = axes[0, 1]
            if metrics['probabilities'] is not None:
                max_proba = np.max(metrics['probabilities'], axis=1)
                ax2.hist(max_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Prediction Confidence Distribution')
                ax2.set_xlabel('Maximum Probability')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No probability data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Prediction Confidence Distribution')
            
            # 3. Class-wise performance
            ax3 = axes[1, 0]
            class_metrics = {
                'Precision': [metrics['precision_macro'], metrics['precision_weighted']],
                'Recall': [metrics['recall_macro'], metrics['recall_weighted']],
                'F1-Score': [metrics['f1_macro'], metrics['f1_weighted']]
            }
            
            x = np.arange(len(class_metrics))
            width = 0.35
            
            ax3.bar(x - width/2, [class_metrics[k][0] for k in class_metrics.keys()], 
                   width, label='Macro', alpha=0.8)
            ax3.bar(x + width/2, [class_metrics[k][1] for k in class_metrics.keys()], 
                   width, label='Weighted', alpha=0.8)
            
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Score')
            ax3.set_title('Class-wise Performance Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(class_metrics.keys())
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Performance summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = f"""
Performance Summary for {model_name}

Accuracy: {metrics['accuracy']:.4f}
F1-Score (Macro): {metrics['f1_macro']:.4f}
F1-Score (Weighted): {metrics['f1_weighted']:.4f}
Precision (Macro): {metrics['precision_macro']:.4f}
Recall (Macro): {metrics['recall_macro']:.4f}
ROC AUC: {metrics['roc_auc']:.4f if metrics['roc_auc'] else 'N/A'}

Model Type: {'Supervised' if 'rf_model' in model_name.lower() else 'Unsupervised'}
            """
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{model_name.replace(" ", "_").lower()}_detailed.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Individual model plots saved to: {self.output_dir}/")
    
    def create_confusion_matrices(self):
        """Create confusion matrices for all models."""
        print("Creating confusion matrices...")
        
        n_models = len(self.metrics)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(6 * ((n_models + 1) // 2), 10))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(self.metrics.items()):
            cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Normal', 'ARP Spoof', 'Other'],
                       yticklabels=['Normal', 'ARP Spoof', 'Other'])
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to: {self.output_dir}/confusion_matrices.png")
        plt.show()
    
    def create_roc_curves(self):
        """Create ROC curves for models that support it."""
        print("Creating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        for model_name, metrics in self.metrics.items():
            if metrics['probabilities'] is not None and metrics['roc_auc'] is not None:
                if metrics['probabilities'].shape[1] == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['probabilities'][:, 1])
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
                else:
                    # Multi-class classification
                    fpr, tpr, _ = roc_curve(metrics['true_labels'], 
                                          metrics['probabilities'].max(axis=1))
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {self.output_dir}/roc_curves.png")
        plt.show()
    
    def create_precision_recall_curves(self):
        """Create Precision-Recall curves."""
        print("Creating Precision-Recall curves...")
        
        plt.figure(figsize=(12, 8))
        
        for model_name, metrics in self.metrics.items():
            if metrics['probabilities'] is not None:
                if metrics['probabilities'].shape[1] == 2:
                    # Binary classification
                    precision, recall, _ = precision_recall_curve(
                        metrics['true_labels'], metrics['probabilities'][:, 1])
                    plt.plot(recall, precision, label=f'{model_name}')
                else:
                    # Multi-class classification
                    precision, recall, _ = precision_recall_curve(
                        metrics['true_labels'], metrics['probabilities'].max(axis=1))
                    plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Precision-Recall curves saved to: {self.output_dir}/precision_recall_curves.png")
        plt.show()
    
    def create_feature_importance_plots(self):
        """Create feature importance plots for Random Forest models."""
        print("Creating feature importance plots...")
        
        rf_models = {name: data for name, data in self.models.items() 
                    if 'random' in name.lower() or 'rf' in name.lower()}
        
        if not rf_models:
            print("No Random Forest models found for feature importance analysis")
            return
        
        n_models = len(rf_models)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(8 * ((n_models + 1) // 2), 12))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, model_data) in enumerate(rf_models.items()):
            if 'feature_names' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                feature_importance = model_data['model'].feature_importances_
                feature_names = model_data['feature_names']
                
                # Create DataFrame for easier plotting
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)
                
                # Plot top 15 features
                top_features = importance_df.tail(15)
                axes[i].barh(range(len(top_features)), top_features['importance'])
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features['feature'])
                axes[i].set_title(f'{model_name}\nTop 15 Feature Importance')
                axes[i].set_xlabel('Importance')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(rf_models), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plots saved to: {self.output_dir}/feature_importance.png")
        plt.show()
    
    def create_performance_summary_table(self):
        """Create a performance summary table."""
        print("Creating performance summary table...")
        
        # Prepare data for table
        summary_data = []
        for model_name, metrics in self.metrics.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Score (Macro)': f"{metrics['f1_macro']:.4f}",
                'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}",
                'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
                'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A'
            })
        
        df = pd.DataFrame(summary_data)
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/performance_summary_table.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Performance summary table saved to: {self.output_dir}/performance_summary_table.png")
        plt.show()
        
        # Also save as CSV
        df.to_csv(f'{self.output_dir}/performance_summary.csv', index=False)
        print(f"✅ Performance summary CSV saved to: {self.output_dir}/performance_summary.csv")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("="*60)
        print("COMPREHENSIVE MODEL METRICS VISUALIZATION GENERATOR")
        print("="*60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models found to evaluate!")
            return
        
        # Generate all visualizations
        self.create_comprehensive_metrics_dashboard()
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 All outputs saved to: {self.output_dir}/")
        print("\nGenerated files:")
        
        output_files = [
            "model_comparison.png",
            "confusion_matrices.png", 
            "roc_curves.png",
            "precision_recall_curves.png",
            "feature_importance.png",
            "performance_summary_table.png",
            "performance_summary.csv"
        ]
        
        for file in output_files:
            if os.path.exists(f"{self.output_dir}/{file}"):
                print(f"  ✅ {file}")
        
        # Individual model plots
        for model_name in self.models.keys():
            individual_file = f"{model_name.replace(' ', '_').lower()}_detailed.png"
            if os.path.exists(f"{self.output_dir}/{individual_file}"):
                print(f"  ✅ {individual_file}")
        
        print(f"\n🎯 Key Metrics Available:")
        print(f"  • F1 Score (Macro & Weighted)")
        print(f"  • Accuracy")
        print(f"  • Precision & Recall")
        print(f"  • ROC AUC")
        print(f"  • Confusion Matrices")
        print(f"  • Feature Importance")
        print(f"  • Performance Comparisons")

def main():
    """Main function to generate all visualizations."""
    visualizer = ModelMetricsVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 