# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

# Alternative classifiers
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Load Dataset
df = pd.read_csv('news.csv')

# Step 3: Remove Unwanted Column
dataset = df.drop("Unnamed: 0", axis=1)

# Step 4: Separate Input and Output
X = dataset['text']
y = dataset['label']

# Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=53
)

# Step 6: Apply Hashing Vectorizer
# alternate_sign=False keeps values non-negative (needed for ComplementNB)
hash_vectorizer = HashingVectorizer(
    stop_words='english',
    n_features=2**18,
    alternate_sign=False
)

# HashingVectorizer is stateless, so no fit is needed
hash_train = hash_vectorizer.transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

# Step 7: Define the models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50, random_state=53),
    "Linear SVM": LinearSVC(),
    "SGD Classifier": SGDClassifier(random_state=53),
    "Complement NB": ComplementNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=53, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=53),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric='cosine'),
}

# Step 8: Train, predict, and evaluate each model
results = {}
for name, model in models.items():
    model.fit(hash_train, y_train)
    y_pred = model.predict(hash_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    results[name] = acc

    print("=" * 60)
    print(f"{name}  |  Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred, labels=model.classes_))
    print(classification_report(y_test, y_pred))

# Step 9: Compare all models
print("=" * 60)
print("Model comparison (sorted by accuracy):")
for name, acc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{name:<22} {acc:.3f}")

# Step 10: Test the best model on a new article
best_name = max(results, key=results.get)
best_model = models[best_name]
sample = ["Breaking: scientists announce a new discovery about the moon."]
sample_hash = hash_vectorizer.transform(sample)
print(f"\nBest model: {best_name}")
print("Prediction:", best_model.predict(sample_hash)[0])
