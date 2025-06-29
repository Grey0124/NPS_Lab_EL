#!/usr/bin/env python3
"""
Feature Analysis for ARP Spoofing Detection Model
This script analyzes feature importance and potential data leakage issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load dataset and perform initial analysis."""
    print("Loading dataset...")
    df = pd.read_csv('../dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Check for potential ID/sequence features
    id_features = ['frame.number', 'tcp.seq']
    print(f"\n🔍 Checking potential ID features: {id_features}")
    
    for feature in id_features:
        if feature in df.columns:
            unique_ratio = df[feature].nunique() / len(df)
            print(f"  {feature}: {unique_ratio:.4f} unique ratio")
            if unique_ratio > 0.9:
                print(f"    ⚠️  WARNING: {feature} appears to be an ID/sequence feature!")
    
    return df

def analyze_feature_importance(df):
    """Analyze feature importance with different feature sets."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with all features
    print("Training RandomForest with all features...")
    rf_all = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_all.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importance_all = rf_all.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_all
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Feature Importance (All Features):")
    for idx, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Identify suspicious features
    suspicious_features = []
    for feature in ['frame.number', 'tcp.seq']:
        if feature in feature_cols:
            idx = feature_cols.index(feature)
            importance = importance_all[idx]
            if importance > 0.1:  # High importance threshold
                suspicious_features.append((feature, importance))
    
    if suspicious_features:
        print(f"\n⚠️  SUSPICIOUS FEATURES (High Importance):")
        for feature, importance in suspicious_features:
            print(f"  {feature}: {importance:.4f}")
    
    return feature_importance_df, suspicious_features

def test_without_suspicious_features(df, suspicious_features):
    """Test model performance without suspicious features."""
    if not suspicious_features:
        print("\n✅ No suspicious features identified.")
        return
    
    print("\n" + "="*60)
    print("TESTING WITHOUT SUSPICIOUS FEATURES")
    print("="*60)
    
    # Remove suspicious features
    suspicious_feature_names = [f[0] for f in suspicious_features]
    clean_features = [col for col in df.columns if col != 'label' and col not in suspicious_feature_names]
    
    print(f"Removing features: {suspicious_feature_names}")
    print(f"Remaining features: {len(clean_features)}")
    
    # Prepare clean data
    X_clean = df[clean_features].values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model without suspicious features
    print("Training RandomForest without suspicious features...")
    rf_clean = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_clean.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_clean.predict(X_test_scaled)
    
    print("\n📊 Performance Without Suspicious Features:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in label_encoder.classes_]))
    
    return rf_clean, clean_features

def analyze_data_leakage(df):
    """Analyze potential data leakage issues."""
    print("\n" + "="*60)
    print("DATA LEAKAGE ANALYSIS")
    print("="*60)
    
    # Check for temporal leakage
    if 'frame.number' in df.columns:
        print("🔍 Checking for temporal leakage...")
        
        # Sort by frame number and check label distribution
        df_sorted = df.sort_values('frame.number')
        
        # Check if labels are clustered by frame number
        label_changes = (df_sorted['label'] != df_sorted['label'].shift()).sum()
        total_frames = len(df_sorted)
        
        print(f"  Total frames: {total_frames}")
        print(f"  Label changes: {label_changes}")
        print(f"  Change ratio: {label_changes/total_frames:.4f}")
        
        if label_changes/total_frames < 0.01:
            print("    ⚠️  WARNING: Very few label changes - possible temporal clustering!")
    
    # Check for feature-label correlation
    print("\n🔍 Checking feature-label correlations...")
    feature_cols = [col for col in df.columns if col != 'label']
    
    correlations = []
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            corr = abs(df[feature].corr(df['label']))
            correlations.append((feature, corr))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("  Top 5 feature-label correlations:")
    for feature, corr in correlations[:5]:
        print(f"    {feature}: {corr:.4f}")
        if corr > 0.8:
            print(f"      ⚠️  WARNING: Very high correlation - possible leakage!")

def create_visualizations(df, feature_importance_df):
    """Create visualizations for analysis."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ARP Spoofing Detection - Feature Analysis', fontsize=16)
    
    # 1. Feature Importance
    top_features = feature_importance_df.head(10)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Class Distribution
    label_counts = df['label'].value_counts()
    axes[0, 1].bar(label_counts.index, label_counts.values, color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Class Distribution')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature correlations heatmap (top features)
    top_feature_names = top_features['feature'].tolist()
    if len(top_feature_names) > 1:
        corr_matrix = df[top_feature_names].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], square=True)
        axes[1, 0].set_title('Top Features Correlation Matrix')
    
    # 4. Feature distributions by class
    if 'arp.opcode' in df.columns:
        df.boxplot(column='arp.opcode', by='label', ax=axes[1, 1])
        axes[1, 1].set_title('ARP Opcode Distribution by Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('ARP Opcode')
    
    plt.tight_layout()
    plt.savefig('models/feature_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to: models/feature_analysis.png")

def main():
    """Main analysis function."""
    print("ARP Spoofing Detection - Feature Analysis")
    print("="*60)
    
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Analyze feature importance
    feature_importance_df, suspicious_features = analyze_feature_importance(df)
    
    # Test without suspicious features
    if suspicious_features:
        test_without_suspicious_features(df, suspicious_features)
    
    # Analyze data leakage
    analyze_data_leakage(df)
    
    # Create visualizations
    create_visualizations(df, feature_importance_df)
    
    print("\n" + "="*60)
    print("✅ Feature analysis completed!")
    print("📊 Check models/feature_analysis.png for visualizations")
    print("="*60)

if __name__ == "__main__":
    main() 