#!/usr/bin/env python3
"""
Download Taiwan Credit Card Default dataset from UCI ML Repository
Dataset: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def download_dataset():
    """Download and save the Taiwan Credit Card dataset."""
    print("Downloading Taiwan Credit Card Default dataset...")
    
    # Fetch dataset from UCI ML Repository
    # ID 350: Default of Credit Card Clients Dataset
    dataset = fetch_ucirepo(id=350)
    
    # Get features and target
    X = dataset.data.features
    y = dataset.data.targets
    
    # Combine into single dataframe
    df = pd.concat([X, y], axis=1)
    
    # Display info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Target column: {y.columns.tolist()}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    # Save to data folder
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/credit_card_default.csv', index=False)
    print(f"\n✅ Dataset saved to data/credit_card_default.csv")
    
    # Save metadata
    with open('data/dataset_info.txt', 'w') as f:
        f.write(f"Dataset: Default of Credit Card Clients\n")
        f.write(f"Source: UCI ML Repository (ID: 350)\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Target: {y.columns.tolist()}\n")
        f.write(f"\nFeature names:\n")
        for col in X.columns:
            f.write(f"  - {col}\n")
    
    print("✅ Metadata saved to data/dataset_info.txt")
    
    return df

if __name__ == "__main__":
    df = download_dataset()
    print("\n" + "="*50)
    print("Sample data:")
    print(df.head())
