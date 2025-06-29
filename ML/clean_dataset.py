#!/usr/bin/env python3
"""
Dataset Cleaning for ARP Spoofing Detection
Remove data leakage sources and create clean dataset.
"""

import pandas as pd
import os
from datetime import datetime

def clean_dataset():
    """Clean the dataset by removing data leakage sources."""
    print("ARP Spoofing Detection - Dataset Cleaning")
    print("="*60)
    
    # Load original dataset
    input_path = '../dataset.csv'
    print(f"Loading dataset from: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Features to exclude (data leakage sources)
    excluded_features = [
        'frame.number',          # Perfect mapping to labels
        'data.len',              # High correlation (0.7210)
        'tcp.flag_syn',          # High correlation (0.5203)
        'tcp.seq',               # High deterministic relationship
        'arp.opcode',            # Deterministic mapping per class
        'arp.src.hw_mac',        # Deterministic mapping per class
        'arp.dst.hw_mac'         # Deterministic mapping per class
    ]
    
    # Features to keep (genuine network behavior)
    clean_features = [
        'frame.time_delta',      # Time between packets
        'tcp.hdr_len',           # TCP header length
        'tcp.flag_ack',          # TCP ACK flag
        'tcp.flag_psh',          # TCP PSH flag
        'tcp.flag_rst',          # TCP RST flag
        'tcp.flag_fin',          # TCP FIN flag
        'icmp.type'              # ICMP type
    ]
    
    # Check available features
    available_clean = [f for f in clean_features if f in df.columns]
    present_excluded = [f for f in excluded_features if f in df.columns]
    
    print(f"Features to keep: {available_clean}")
    print(f"Features to exclude: {present_excluded}")
    
    # Create clean dataset
    clean_columns = available_clean + ['label']
    df_clean = df[clean_columns].copy()
    
    print(f"Clean dataset shape: {df_clean.shape}")
    
    # Analyze correlations in clean dataset
    print(f"\nFeature correlations in clean dataset:")
    for feature in available_clean:
        if df_clean[feature].dtype in ['int64', 'float64']:
            corr = abs(df_clean[feature].corr(df_clean['label']))
            print(f"  {feature}: {corr:.4f}")
    
    # Save clean dataset
    output_path = '../dataset_clean.csv'
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ Clean dataset saved to: {output_path}")
    print(f"📊 Original: {df.shape} -> Clean: {df_clean.shape}")
    
    return output_path

if __name__ == "__main__":
    clean_dataset() 