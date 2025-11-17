import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv('Data/phishing_legit_dataset_KD_10000_utf8.csv')  # Update path if needed

# Use correct column names: 'text_combined' for email text and 'label' for labels
X = df['text_combined'].str.lower()
y = df['label']

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

# Display results and best model
best_model = max(results, key=lambda x: results[x]['accuracy'])
print("\nModel Comparison Results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-Score={metrics['f1-score']:.4f}")

print(f"\nBest performing model based on accuracy: {best_model}")