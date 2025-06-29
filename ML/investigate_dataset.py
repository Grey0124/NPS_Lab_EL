#!/usr/bin/env python3
"""
Dataset Investigation for ARP Spoofing Detection
This script investigates the dataset creation process and labeling methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_examine_dataset(filepath):
    """Load and perform initial examination of the dataset."""
    print("Loading and examining dataset...")
    
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    
    return df

def analyze_label_distribution(df):
    """Analyze the distribution and patterns of labels."""
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Label counts
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Label {label}: {count:,} samples ({percentage:.2f}%)")
    
    # Check for label patterns
    print(f"\nLabel pattern analysis:")
    
    # Check if labels are clustered
    label_changes = (df['label'] != df['label'].shift()).sum()
    total_samples = len(df)
    change_ratio = label_changes / total_samples
    
    print(f"  Total label changes: {label_changes}")
    print(f"  Change ratio: {change_ratio:.4f}")
    
    if change_ratio < 0.01:
        print("    ⚠️  WARNING: Very few label changes - possible temporal clustering!")
    elif change_ratio > 0.5:
        print("    ✅ Good label distribution - frequent changes")
    else:
        print("    ⚠️  Moderate clustering - investigate further")
    
    return label_counts

def investigate_feature_label_relationships(df):
    """Investigate relationships between features and labels."""
    print("\n" + "="*60)
    print("FEATURE-LABEL RELATIONSHIP ANALYSIS")
    print("="*60)
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    # Calculate correlations
    correlations = []
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            corr = abs(df[feature].corr(df['label']))
            correlations.append((feature, corr))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature-Label Correlations (sorted by strength):")
    for feature, corr in correlations:
        print(f"  {feature}: {corr:.4f}")
        if corr > 0.8:
            print(f"    🚨 CRITICAL: Very high correlation - likely data leakage!")
        elif corr > 0.5:
            print(f"    ⚠️  WARNING: High correlation - investigate!")
        elif corr > 0.3:
            print(f"    📊 Moderate correlation")
        else:
            print(f"    ✅ Low correlation")
    
    return correlations

def analyze_feature_distributions_by_label(df):
    """Analyze how feature distributions differ across labels."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION BY LABEL")
    print("="*60)
    
    # Focus on key features
    key_features = ['arp.opcode', 'arp.src.hw_mac', 'arp.dst.hw_mac', 'data.len', 'tcp.flag_syn']
    
    for feature in key_features:
        if feature in df.columns:
            print(f"\n{feature} distribution by label:")
            
            # Group by label and analyze
            for label in sorted(df['label'].unique()):
                label_data = df[df['label'] == label][feature]
                
                if label_data.dtype in ['int64', 'float64']:
                    print(f"  Label {label}:")
                    print(f"    Count: {len(label_data)}")
                    print(f"    Unique values: {label_data.nunique()}")
                    print(f"    Mean: {label_data.mean():.4f}")
                    print(f"    Std: {label_data.std():.4f}")
                    print(f"    Range: [{label_data.min()}, {label_data.max()}]")
                else:
                    # For categorical features
                    value_counts = label_data.value_counts()
                    print(f"  Label {label}:")
                    print(f"    Count: {len(label_data)}")
                    print(f"    Unique values: {label_data.nunique()}")
                    print(f"    Top 3 values: {value_counts.head(3).to_dict()}")

def investigate_temporal_patterns(df):
    """Investigate temporal patterns in the data."""
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    
    if 'frame.number' in df.columns:
        print("Analyzing frame number patterns...")
        
        # Sort by frame number
        df_sorted = df.sort_values('frame.number')
        
        # Check for label clustering by frame number
        label_changes = (df_sorted['label'] != df_sorted['label'].shift()).sum()
        total_frames = len(df_sorted)
        
        print(f"  Total frames: {total_frames}")
        print(f"  Label changes: {label_changes}")
        print(f"  Change ratio: {label_changes/total_frames:.4f}")
        
        # Check for large continuous blocks
        current_label = None
        current_count = 0
        max_continuous = 0
        
        for label in df_sorted['label']:
            if label == current_label:
                current_count += 1
            else:
                if current_count > max_continuous:
                    max_continuous = current_count
                current_label = label
                current_count = 1
        
        print(f"  Maximum continuous same-label block: {max_continuous}")
        print(f"  Average block size: {total_frames / label_changes:.1f}")
        
        if max_continuous > total_frames * 0.1:
            print("    ⚠️  WARNING: Large continuous blocks detected!")
    
    if 'frame.time_delta' in df.columns:
        print("\nAnalyzing time delta patterns...")
        
        # Check if time deltas differ by label
        for label in sorted(df['label'].unique()):
            label_data = df[df['label'] == label]['frame.time_delta']
            print(f"  Label {label} time delta: mean={label_data.mean():.6f}, std={label_data.std():.6f}")

def check_for_data_leakage_indicators(df):
    """Check for various data leakage indicators."""
    print("\n" + "="*60)
    print("DATA LEAKAGE INDICATOR ANALYSIS")
    print("="*60)
    
    leakage_indicators = []
    
    # 1. Check for perfect feature-label mappings
    print("Checking for perfect feature-label mappings...")
    feature_cols = [col for col in df.columns if col != 'label']
    
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            # Check if feature perfectly predicts label
            unique_combinations = df[[feature, 'label']].drop_duplicates()
            if len(unique_combinations) == len(df[feature].unique()):
                leakage_indicators.append(f"Perfect mapping: {feature} -> label")
                print(f"  🚨 CRITICAL: {feature} perfectly maps to labels!")
    
    # 2. Check for deterministic relationships
    print("\nChecking for deterministic relationships...")
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            # Group by feature and check label consistency
            feature_label_groups = df.groupby(feature)['label'].agg(['count', 'nunique'])
            perfect_predictions = (feature_label_groups['nunique'] == 1).sum()
            total_groups = len(feature_label_groups)
            
            if perfect_predictions / total_groups > 0.8:
                leakage_indicators.append(f"High deterministic relationship: {feature}")
                print(f"  ⚠️  WARNING: {feature} has high deterministic relationship with labels")
    
    # 3. Check for synthetic patterns
    print("\nChecking for synthetic data patterns...")
    
    # Check for too-perfect distributions
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            # Check if values are too evenly distributed
            value_counts = df[feature].value_counts()
            if len(value_counts) > 10:  # Only for features with many values
                # Calculate entropy-like measure
                proportions = value_counts / len(df)
                entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
                max_entropy = np.log2(len(value_counts))
                
                if entropy / max_entropy > 0.9:
                    print(f"  📊 {feature}: Very uniform distribution (entropy ratio: {entropy/max_entropy:.3f})")
    
    return leakage_indicators

def create_investigation_visualizations(df):
    """Create visualizations for dataset investigation."""
    print("\nCreating investigation visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ARP Spoofing Dataset Investigation', fontsize=16)
    
    # 1. Label distribution
    label_counts = df['label'].value_counts().sort_index()
    axes[0, 0].bar(label_counts.index, label_counts.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Label Distribution')
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature-label correlations
    feature_cols = [col for col in df.columns if col != 'label']
    correlations = []
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            corr = abs(df[feature].corr(df['label']))
            correlations.append((feature, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = correlations[:10]
    
    features, corrs = zip(*top_features)
    axes[0, 1].barh(range(len(features)), corrs)
    axes[0, 1].set_yticks(range(len(features)))
    axes[0, 1].set_yticklabels(features)
    axes[0, 1].set_title('Top 10 Feature-Label Correlations')
    axes[0, 1].set_xlabel('Absolute Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ARP opcode distribution by label
    if 'arp.opcode' in df.columns:
        df.boxplot(column='arp.opcode', by='label', ax=axes[0, 2])
        axes[0, 2].set_title('ARP Opcode Distribution by Label')
        axes[0, 2].set_xlabel('Label')
        axes[0, 2].set_ylabel('ARP Opcode')
    
    # 4. Data length distribution by label
    if 'data.len' in df.columns:
        df.boxplot(column='data.len', by='label', ax=axes[1, 0])
        axes[1, 0].set_title('Data Length Distribution by Label')
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Data Length')
    
    # 5. Label changes over time (if frame.number exists)
    if 'frame.number' in df.columns:
        df_sorted = df.sort_values('frame.number')
        label_changes = (df_sorted['label'] != df_sorted['label'].shift()).cumsum()
        axes[1, 1].plot(df_sorted['frame.number'], label_changes, alpha=0.7)
        axes[1, 1].set_title('Label Changes Over Time')
        axes[1, 1].set_xlabel('Frame Number')
        axes[1, 1].set_ylabel('Cumulative Label Changes')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature correlation heatmap (top features)
    top_feature_names = [f[0] for f in top_features[:8]]
    if len(top_feature_names) > 1:
        corr_matrix = df[top_feature_names + ['label']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 2], square=True, fmt='.2f')
        axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('models/dataset_investigation.png', dpi=300, bbox_inches='tight')
    print("Investigation visualizations saved to: models/dataset_investigation.png")

def generate_investigation_report(df, label_counts, correlations, leakage_indicators):
    """Generate a comprehensive investigation report."""
    print("\n" + "="*60)
    print("DATASET INVESTIGATION REPORT")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Classes: {len(label_counts)}")
    
    print(f"\n🏷️  Label Analysis:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {label}: {count:,} samples ({percentage:.2f}%)")
    
    print(f"\n🔗 Feature-Label Relationships:")
    high_corr_features = [f for f, c in correlations if c > 0.5]
    if high_corr_features:
        print(f"  High correlation features (>0.5): {high_corr_features}")
    else:
        print(f"  No features with correlation > 0.5")
    
    print(f"\n🚨 Data Leakage Indicators:")
    if leakage_indicators:
        for indicator in leakage_indicators:
            print(f"  ⚠️  {indicator}")
    else:
        print(f"  ✅ No obvious data leakage indicators found")
    
    print(f"\n📋 Recommendations:")
    if len(high_corr_features) > 0:
        print(f"  1. Investigate high correlation features: {high_corr_features}")
    if leakage_indicators:
        print(f"  2. Address data leakage issues before model deployment")
    print(f"  3. Test model on completely independent network captures")
    print(f"  4. Consider using only basic ARP features for initial testing")
    
    print("="*60)

def main():
    """Main investigation function."""
    print("ARP Spoofing Detection - Dataset Investigation")
    print("="*60)
    
    try:
        # Load dataset
        dataset_path = '../dataset.csv'
        df = load_and_examine_dataset(dataset_path)
        
        # Perform analyses
        label_counts = analyze_label_distribution(df)
        correlations = investigate_feature_label_relationships(df)
        analyze_feature_distributions_by_label(df)
        investigate_temporal_patterns(df)
        leakage_indicators = check_for_data_leakage_indicators(df)
        
        # Create visualizations
        create_investigation_visualizations(df)
        
        # Generate report
        generate_investigation_report(df, label_counts, correlations, leakage_indicators)
        
        print(f"\n✅ Dataset investigation completed!")
        print(f"📊 Check models/dataset_investigation.png for visualizations")
        
    except Exception as e:
        print(f"❌ Error during investigation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 