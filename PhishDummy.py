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
features = ['Phishing_Type', 'Severity', 'Confidence']

df['Serverity'] = df['Severity'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0)
df['Phishing_Type'] = df['Phishing_Type'].map({'Legitimate': 0, 'Phishing': 1}).fillna(0)
df['Confidence'] = df['Confidence'].fillna(0).astype(float)

y = df['Serverity'].values

print( "[OK] Labels shape: " + str(y.shape) + "\n")
