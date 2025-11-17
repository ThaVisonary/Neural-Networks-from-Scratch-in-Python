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

print("[OK] Features shape: " + str(y.shape) + "\n")
print("Class: 0 = Legitimate, 1 = Phishing", np.bincount(y) + "\n")
