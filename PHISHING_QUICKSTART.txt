# 🎯 PHISHING DETECTION: Quick Start

## Your Dataset
- **Files:** CSV file
- **Features:** 4 numeric (Detected, Malicious, Suspicious, Malware)
- **Task:** Binary classification (legitimate vs phishing)

---

## 🚀 Run Training NOW

```bash
python 'encode_phishing_dataset'
python 'PhishingTSample'
```

That's it! The script will:
1. ✅ Load your dataset
2. ✅ Normalize features
3. ✅ Split into 80% train, 20% test
4. ✅ Train neural network (100 iterations)
5. ✅ Evaluate accuracy
6. ✅ Show loss curves
7. ✅ Save model (optional)

---

## 📊 What You'll See

```
Training progress → Shows which iteration it's on
Loss decreasing → Model is learning
Final Accuracy → % of correct predictions
Plots → Visual results
```

Expected output example:
```
✅ Test Accuracy: 85.32%
   Correct: 181/212

Class 0 (Legitimate): 92.5%
Class 1 (Phishing):   78.3%
```

---

Steps:
1. Upload dataset into Data folder
2. Encode the dataset in encode_phishing_dataset.py
3. Run the dataset in PhishingTSample 
