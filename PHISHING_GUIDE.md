# 🎯 How to Train Neural Network for Phishing Detection

## Your Dataset Summary

**Dataset:** `phishing_dataset.csv`
- **Total Samples:** 1,062 phishing/spam messages
- **Total Features:** 23 columns
- **Key Columns for Classification:**
  - `Phishing` - Phishing detection label (0-18 values)
  - `Detected` - Whether message was detected as threat
  - `Malicious` - Whether message is malicious
  - `Suspicious` - Whether message is suspicious
  - `Malware` - Whether message contains malware

**Numeric Features Available:**
1. Detected (0 or 1)
2. Malicious (0 or 1)
3. Phishing (0-18, multi-class)
4. Suspicious (0 or 1)
5. Malware (0 or 1)

---

## 🚀 How to Train

### Step 1: Prepare Your Data

Your dataset is already in the correct format and location:
```
data/phishing_clean.csv  (UTF-8 encoded version)
```

### Step 2: Create a Simple CSV with Binary Labels

For easier training, create a simplified CSV with just the key features:

```python
import pandas as pd
import numpy as np

# Load original data
df = pd.read_csv('data/phishing_clean.csv')

# Extract numeric features
features = ['Detected', 'Malicious', 'Suspicious', 'Malware']
X = df[features].fillna(0).astype(int)

# Create binary label: 1 if any threat detected, 0 otherwise
y = ((df['Phishing'] > 0) | (df['Malicious'] > 0) | (df['Detected'] > 0)).astype(int)

# Save simplified dataset
output_df = pd.concat([X, pd.DataFrame({'is_phishing': y})], axis=1)
output_df.to_csv('data/phishing_binary.csv', index=False)

print(f"Created simplified dataset:")
print(f"  Samples: {len(output_df)}")
print(f"  Class 0 (Legitimate): {(y==0).sum()}")
print(f"  Class 1 (Phishing): {(y==1).sum()}")
```

### Step 3: Use the Training Script

```bash
python train_phishing.py
```

---

## 📊 Understanding Your Phishing Data

### Features:
- **Detected**: Binary (0/1) - Was threat detected by system?
- **Malicious**: Binary (0/1) - Is message malicious?
- **Suspicious**: Binary (0/1) - Is message suspicious?
- **Malware**: Binary (0/1) - Does message contain malware?
- **Phishing**: Multi-class (0-18) - Type of phishing/threat

### Label Distribution (from 1,062 samples):
- Negative (0): 559 samples (52.6%)
- Positive (1-18): 503 samples (47.4%)

This is a **balanced dataset** - good for training!

---

## 🛠️ Two Approaches

### Approach 1: Binary Classification (Easier)
Train to detect: **Phishing or Not?**
- **Input:** 4 numeric features (Detected, Malicious, Suspicious, Malware)
- **Output:** 2 classes (0=legitimate, 1=phishing)
- **Best for:** Quick detection

### Approach 2: Multi-class Classification (Advanced)
Train to classify: **Type of threat**
- **Input:** 4 numeric features
- **Output:** 19 classes (threat types)
- **Best for:** Detailed threat analysis

---

## 💻 Ready-to-Use Training Commands

### Quick Start
```bash
# The training script is already created
python train_phishing.py
```

### Manual Training (if needed)
```python
import pandas as pd
import numpy as np
from Layers import Helpers
from Models.LeNet import build

# Load data
df = pd.read_csv('data/phishing_clean.csv')

# Prepare features
X = df[['Detected', 'Malicious', 'Suspicious', 'Malware']].fillna(0).values
y = ((df['Phishing'] > 0) | (df['Malicious'] > 0)).astype(int).values

# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
X = (X - X.min()) / (X.max() - X.min() + 1e-8)

# Create one-hot labels
y_one_hot = np.zeros((len(y), 2))
for i in range(len(y)):
    y_one_hot[i, int(y[i])] = 1

# Split data (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_one_hot[:split], y_one_hot[split:]

print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")
```

---

## 📈 Expected Results

With your dataset and the neural network:

- **Training Samples:** ~850 (80%)
- **Test Samples:** ~212 (20%)
- **Expected Accuracy:** 70-95% (depends on model complexity)
- **Training Time:** 30-60 seconds for 100 iterations

---

## ✅ What Happens During Training

1. **Data Loading** → Reads phishing dataset
2. **Normalization** → Scales features to [0, 1] range
3. **Splitting** → 80% training, 20% testing
4. **Training** → Neural network learns patterns (100 iterations)
5. **Evaluation** → Tests on unseen data
6. **Visualization** → Shows loss curves
7. **Saving** → Saves trained model (optional)

---

## 🎯 Performance Metrics

The neural network will report:
- **Loss:** How well it learned (lower is better)
- **Accuracy:** % of correct predictions
- **Per-class accuracy:** Accuracy for each threat type

Example output:
```
✅ Test Accuracy: 85.32%
   Correct: 181/212

Training Loss:  [starts high] → [decreases] → [stabilizes]
Test Accuracy:  ~65% → ~85% (improves over iterations)
```

---

## 📝 Next Steps

1. **Run the training script:**
   ```bash
   python train_phishing.py
   ```

2. **Monitor the output** for:
   - Data loading confirmation
   - Training progress
   - Final accuracy
   - Loss visualization

3. **Save the model** when prompted

4. **Use the model** for phishing detection on new messages

---

## 🔍 Interpreting Results

**High Accuracy (85%+):** Model learned phishing patterns well
**Medium Accuracy (60-85%):** Model is learning, needs more training
**Low Accuracy (<60%):** Model needs more data or feature engineering

Your dataset is good size and quality for training!

---

## 📚 File Locations

```
Neural-Networks-from-Scratch-in-Python/
├── data/
│   ├── phishing_dataset.csv      (Original, encoding issues)
│   ├── phishing_clean.csv         (Fixed encoding, ready to use)
│   └── phishing_binary.csv        (Optional: simplified version)
├── train_phishing.py              (Main training script)
├── analyze_phishing.py            (Data analysis tool)
└── trained/
    └── phishing_model             (Saved model after training)
```

---

## 💡 Tips for Better Results

1. **Increase iterations:** Change 100 to 200-300
2. **Adjust batch size:** Try 16, 32, or 64
3. **Add more features:** Use additional columns if meaningful
4. **Feature engineering:** Create new features from existing ones
5. **Data augmentation:** If dataset seems small

---

## 🎓 Understanding Phishing Detection

Your neural network will learn:
- Patterns in legitimate vs phishing messages
- Feature combinations that indicate threats
- How to classify new messages based on their characteristics

Example features:
- Message from unknown sender → More likely phishing
- Contains suspicious URL → More likely phishing
- Multiple threat flags → Definitely phishing

---

**Ready to train? Run: `python train_phishing.py`** 🚀
