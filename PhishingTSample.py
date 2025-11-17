"""
Phishing Detection - Training Script (Simple ASCII Version)

Trains neural network for phishing detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


print("\n" + "=" * 80)      # Print header
print("PHISHING DETECTION NEURAL NETWORK TRAINING")     # Print title
print("=" * 80 + "\n")      # Print configuration and load data


# Configuration
csv_file = 'Data/testphish.csv'
batch_size = 16
iterations = 100

print("Configuration:")
print("   CSV File: " + csv_file)
print("   Batch Size: " + str(batch_size))
print("   Iterations: " + str(iterations) + "\n")

# Load data
print("Loading dataset...")
try:
    df = pd.read_csv('testphish.csv')
    print("[OK] Loaded " + str(len(df)) + " samples\n")
except Exception as e:
    print("[ERROR] " + str(e) + "\n")
    exit(1)

# Extract features
print("Extracting features...")
feature_cols = ['Detected', 'Malicious', 'Suspicious', 'Malware']
X = df[feature_cols].fillna(0).astype(float).values
print("[OK] Features shape: " + str(X.shape) + "\n")

# Create labels
print("Creating labels...")
y_raw = df['Phishing'].fillna(0) if 'Phishing' in df.columns else df['Malicious'].fillna(0)
y = (y_raw > 0).astype(int).values

class_0_count = (y == 0).sum()
class_1_count = (y == 1).sum()
total = len(y)

print("[OK] Class distribution:")
print("   Legitimate (0): " + str(class_0_count) + " (" + str(round(100*class_0_count/total, 1)) + "%)")
print("   Phishing (1):   " + str(class_1_count) + " (" + str(round(100*class_1_count/total, 1)) + "%)\n")

# Normalize
print("Normalizing features...")
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std
X_min = X_norm.min(axis=0)
X_max = X_norm.max(axis=0)
X_norm = (X_norm - X_min) / (X_max - X_min + 1e-8)
print("[OK] Features normalized to [0,1]\n")

# One-hot encode
print("One-hot encoding labels...")
y_one_hot = np.zeros((len(y), 2))
for i in range(len(y)):
    y_one_hot[i, int(y[i])] = 1
print("[OK] Labels encoded\n")

# Train/test split
print("Splitting data (80/20)...")
split_idx = int(0.8 * len(X_norm))
indices = np.arange(len(X_norm))
np.random.seed(42)
np.random.shuffle(indices)

X_shuffled = X_norm[indices]
y_shuffled = y_one_hot[indices]

X_train = X_shuffled[:split_idx]
y_train = y_shuffled[:split_idx]
X_test = X_shuffled[split_idx:]
y_test = y_shuffled[split_idx:]

print("[OK] Train: " + str(len(X_train)) + ", Test: " + str(len(X_test)) + "\n")

# Summary
print("=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print("\n[OK] Ready for neural network training!\n")

print("Dataset Summary:")
print("   Training samples: " + str(len(X_train)))
print("   Test samples: " + str(len(X_test)))
print("   Input features: " + str(X_train.shape[1]))
print("   Output classes: 2 (Legitimate, Phishing)")
print("   Batch size: " + str(batch_size))
print("   Training iterations: " + str(iterations))

# Now try to import and train
print("\n" + "=" * 80)
print("INITIALIZING NEURAL NETWORK")
print("=" * 80 + "\n")

try:
    from Layers import Helpers
    from Models.LeNet import build
    import NeuralNetwork
    
    print("[OK] Neural network modules imported\n")
    
    # Create data layer
    class PhishingData:
        def __init__(self, X_train, y_train, X_test, y_test, batch_size):
            self.X_train = X_train.copy()
            self.y_train = y_train.copy()
            self.X_test = X_test.copy()
            self.y_test = y_test.copy()
            self.batch_size = batch_size
            self.batch_idx = 0
        
        def next(self):
            start = self.batch_idx * self.batch_size
            end = start + self.batch_size
            
            if end > len(self.X_train):
                idx = np.arange(len(self.X_train))
                np.random.shuffle(idx)
                self.X_train = self.X_train[idx]
                self.y_train = self.y_train[idx]
                self.batch_idx = 0
                start, end = 0, self.batch_size
            
            batch_X = self.X_train[start:end]
            batch_y = self.y_train[start:end]
            self.batch_idx += 1
            
            return batch_X, batch_y
        
        def get_test_set(self):
            return self.X_test, self.y_test
    
    data = PhishingData(X_train, y_train, X_test, y_test, batch_size)
    
    # Build model
    print("Building neural network...")
    net = build()
    net.data_layer = data
    print("[OK] Model built\n")
    
    # Train
    print("Training for " + str(iterations) + " iterations...\n")
    print("=" * 80)
    net.train(iterations)
    print("=" * 80)
    print("\n[OK] Training completed!\n")
    
    # Evaluate
    print("Evaluating on test set...")
    test_data, test_labels = data.get_test_set()
    results = net.test(test_data)
    accuracy = Helpers.calculate_accuracy(results, test_labels)
    
    correct = int(accuracy * len(test_labels))
    total_test = len(test_labels)
    
    print("[OK] Test Accuracy: " + str(round(accuracy * 100, 2)) + "%")
    print("    Correct: " + str(correct) + "/" + str(total_test) + "\n")
    
    # Plot
    print("Generating plot...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(net.loss, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(net.loss) > 1:
        loss_improvement = [net.loss[0] - l for l in net.loss]
        plt.plot(loss_improvement, 'g-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Reduction')
        plt.title('Cumulative Loss Improvement')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print("Initial Loss:    " + str(round(net.loss[0], 6)))
    print("Final Loss:      " + str(round(net.loss[-1], 6)))
    print("Total Reduction: " + str(round(net.loss[0] - net.loss[-1], 6)))
    
    if net.loss[0] > 0:
        improvement = ((net.loss[0] - net.loss[-1]) / net.loss[0]) * 100
        print("Improvement:     " + str(round(improvement, 2)) + "%")
    
    print("Test Accuracy:   " + str(round(accuracy * 100, 2)) + "%")
    print("=" * 80 + "\n")
    
    # Save
    print("Save model? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            os.makedirs('trained', exist_ok=True)
            NeuralNetwork.NeuralNetwork.save('trained/phishing_model', net)
            print("[OK] Model saved to: trained/phishing_model\n")
    except EOFError:
        print("\n[SKIPPED] Model not saved\n")

except ImportError as e:
    print("[WARNING] Could not import neural network modules")
    print("Error: " + str(e) + "\n")
    print("[OK] But your data is prepared!")
    print("To train, install required packages and run again.\n")
except Exception as e:
    print("[ERROR] " + str(e) + "\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("PHISHING DETECTION SETUP COMPLETE")
print("=" * 80 + "\n")
