"""
Phishing Dataset Encoding Fix

This script reads the testphish.csv file with encoding issues and converts it to UTF-8.
The original file uses latin-1 encoding, which causes UTF-8 decode errors.

This is a one-time setup - run once and you can use the cleaned file afterward.
"""

import pandas as pd
import os

print("\n" + "=" * 80)
print("PHISHING DATASET ENCODING CONVERTER")
print("=" * 80 + "\n")

# Input and output paths
input_file = 'Data/url_dataset.csv'
output_file = 'Data/url_dataset_utf8.csv'

print("Input file:  " + input_file)
print("Output file: " + output_file + "\n")

# Step 1: Load with correct encoding
print("Step 1: Reading file with latin-1 encoding...")
try:
    df = pd.read_csv(input_file, encoding='latin-1')
    print("[OK] Loaded " + str(len(df)) + " rows, " + str(len(df.columns)) + " columns\n")
except Exception as e:
    print("[ERROR] Failed to read: " + str(e) + "\n")
    exit(1)

# Step 2: Clean up any remaining encoding issues
print("Step 2: Cleaning encoding issues...")
# Replace any problematic characters
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            # Try to encode as UTF-8 and handle errors
            df[col] = df[col].apply(lambda x: x.encode('utf-8', errors='replace').decode('utf-8') if isinstance(x, str) else x)
        except Exception as e:
            print("[WARNING] Could not clean column '" + col + "': " + str(e))

print("[OK] Encoding cleaned\n")

# Step 3: Save as UTF-8
print("Step 3: Saving as UTF-8...")
try:
    df.to_csv(output_file, index=False, encoding='utf-8')
    print("[OK] File saved\n")
except Exception as e:
    print("[ERROR] Failed to save: " + str(e) + "\n")
    exit(1)

# Step 4: Verify the new file
print("Step 4: Verifying new file...")
try:
    df_test = pd.read_csv(output_file, encoding='utf-8')
    print("[OK] New file reads correctly with UTF-8\n")
except Exception as e:
    print("[ERROR] Verification failed: " + str(e) + "\n")
    exit(1)

# Summary
print("=" * 80)
print("ENCODING COMPLETE")
print("=" * 80)
print("\nDataset Info:")
print("   Rows: " + str(len(df)))
print("   Columns: " + str(len(df.columns)))
print("   File size: " + str(round(os.path.getsize(output_file) / (1024*1024), 2)) + " MB\n")

print("Column Names:")
for i, col in enumerate(df.columns, 1):
    print("   " + str(i) + ". " + col)

print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print("\n1. Update PhishingTSample.py to use:")
print("   csv_file = 'Data/url_dataset_utf8.csv'")
print("\n2. Or use encoding='utf-8' in the script (default)")
print("\n3. Then run: python PhishingTSample.py\n")