# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report

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
# alternate_sign=False keeps all values non-negative (required by MultinomialNB)
# n_features sets the size of the hash space (higher = fewer collisions)
hash_vectorizer = HashingVectorizer(
    stop_words='english',
    n_features=2**18,
    alternate_sign=False
)

# HashingVectorizer is stateless, so no fit is needed
hash_train = hash_vectorizer.transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

# Step 7: Check Hashed Values
print(hash_train)
print("Shape:", hash_train.shape)

# Step 8: Train Machine Learning Model
model = MultinomialNB()
model.fit(hash_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(hash_test)

# Step 10: Evaluate the Model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Step 11: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Confusion Matrix:")
print(cm)

# Step 12: Classification Report
print(classification_report(y_test, y_pred))

# Step 13: Test on a New Article
sample = ["Breaking: scientists announce a new discovery about the moon."]
sample_hash = hash_vectorizer.transform(sample)
print("Prediction:", model.predict(sample_hash)[0])
