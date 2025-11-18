import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n" + "=" * 80)      # Print header
print("PHISHING DETECTION NEURAL NETWORK TRAINING")     # Print title
print("=" * 80 + "\n")      # Print configuration and load data


# Configuration
csv_file = 'Data/phishing_legit_dataset_KD_10000_utf8.csv'
batch_size = 16
iterations = 100

print("Configuration:")
print("   CSV File: " + csv_file)
print("   Batch Size: " + str(batch_size))
print("   Iterations: " + str(iterations) + "\n")

# Load data
print("Loading dataset...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8')
    print("[OK] Loaded " + str(len(df)) + " samples\n")
except Exception as e:
    print("[ERROR] " + str(e) + "\n")
    exit(1)

# Extract features
print("Extracting features...")
features = ['text','label', 'phishing_type', 'severity', 'confidence']
y = ((df['phishing_type'] == 'Phishing') | (df['label'] == 1)).astype(int).values
print("Dataset size: " + str(y.shape) + "\n")
print(f" Class distribution: Legitimate (0): {(y == 0).sum()} ({round(100*(y == 0).sum()/len(y), 1)}%), Phishing (1): {(y == 1).sum()} ({round(100*(y == 1).sum()/len(y), 1)}%)\n")


# Show Sample data
print("Sample data:")

data = {}
for feature in csv_file.split('/')[-1].split('.')[0].split('_'):
    df = pd.read(os.path)

    print(f"{feature} Dataset:")
    print(f"{feature} shape: {df.shape}")
    print(f"{feature} columns: {df.columns.tolist()}")




