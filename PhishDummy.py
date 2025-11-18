import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter



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
features = ['text','label', 'phishing_type', 'severity', 'confidence']          # Define Features
y = ((df['phishing_type'] == 'Phishing') | (df['label'] == 1)).astype(int).values       #Label Creation

# Add the severity to y if it exists in the dataset
if 'severity' in df.columns:
    df['severity'] = df['severity'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0).astype(int)

print(f" Legitimate Emails (0): {(y == 0).sum()} ({round(100*(y == 0).sum()/len(y), 1)}%), Phishing Emails (1): {(y == 1).sum()} ({round(100*(y == 1).sum()/len(y), 1)}%)\n")

# Table of Data Samples and their corresponding labels
print("Sample Data:")
for i in range(5):
    print(f" Sample {i+1}:")
    print(f"  Text: {df['text'].iloc[i][:100]}...")  # Print first 100 characters of the text
    print(f"  Label: {y[i]}\n")
    print(f"  Severity: {df['severity'].iloc[i]}\n")

