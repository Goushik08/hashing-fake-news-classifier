# Fake News Detection with Hashing Vectorizer

Classifies news articles as **FAKE** or **REAL** using scikit-learn's `HashingVectorizer` and compares eight models. Hashing needs no stored vocabulary and uses fixed memory, so it scales to very large or streaming datasets.

**Best model: Linear SVM, 93.1% accuracy** on a held-out test set of 2,091 articles.

## Results

| Rank | Model | Accuracy | FAKE Recall | REAL Recall | Macro F1 |
|---|---|---|---|---|---|
| 1 | Linear SVM | 93.1% | 0.95 | 0.91 | 0.93 |
| 2 | SGD Classifier | 92.6% | 0.93 | 0.92 | 0.93 |
| 3 | Passive Aggressive | 92.1% | 0.93 | 0.92 | 0.92 |
| 4 | Logistic Regression | 90.4% | 0.93 | 0.88 | 0.90 |
| 5 | Random Forest | 89.5% | 0.92 | 0.87 | 0.89 |
| 6 | Complement NB | 85.5% | 0.73 | 0.97 | 0.85 |
| 7 | KNN | 82.9% | 0.71 | 0.94 | 0.83 |
| 8 | Decision Tree | 81.2% | 0.81 | 0.81 | 0.81 |

![Model comparison output](images/model_comparison.png)

Linear SVM confusion matrix (rows = actual, columns = predicted):

| | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actually FAKE** | 959 | 49 |
| **Actually REAL** | 95 | 988 |

Linear models perform best on sparse, high-dimensional text. Complement NB and KNN miss about 27-29% of fake articles despite decent overall accuracy.

## Project Structure

```
hashing-fake-news-classifier/
├── fake_news_classifier.py    # Compares 8 models (main script)
├── baseline_naive_bayes.py    # Simple MultinomialNB baseline
├── requirements.txt
├── images/
│   └── model_comparison.png
└── news.csv                   # You provide this (not included)
```

## Dataset

The scripts expect a file named `news.csv` in the project root with these columns:

| Column | Description |
|---|---|
| `Unnamed: 0` | Index column (dropped by the script) |
| `title` | Article headline |
| `text` | Article body (used as the input) |
| `label` | `FAKE` or `REAL` (the target) |

The results above were produced on a dataset of 6,335 articles, split 67% train / 33% test with `random_state=53`.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/hashing-fake-news-classifier.git
cd hashing-fake-news-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Place `news.csv` in the project folder, then run:

```bash
# Compare all eight models
python fake_news_classifier.py

# Or run only the Naive Bayes baseline
python baseline_naive_bayes.py
```

The comparison script prints accuracy, a confusion matrix and a classification report for every model, then a ranked summary and a prediction on a sample article using the best model.

The code also runs cell by cell in Jupyter Notebook (each `# Step N` comment is one cell).

## How It Works

1. Load `news.csv` and drop the unused index column.
2. Split into train and test sets (33% test).
3. Convert text to numbers with `HashingVectorizer(stop_words='english', n_features=2**18, alternate_sign=False)`. Each word is hashed to one of 262,144 columns, so no vocabulary is stored and no fitting is needed.
4. Train each model on the hashed training matrix.
5. Evaluate on the test set and rank by accuracy.

### Settings worth knowing

- `alternate_sign=False` keeps all values non-negative, which the Naive Bayes models require.
- `n_features=2**18` sets the hash space size. Larger values (for example `2**20`) reduce collisions, where different words share a column, at the cost of more memory.
- Because features are hashed, you cannot map a column back to the original word. Use the TF-IDF version if you need interpretability.

## Limitations

- Results come from a single train/test split.
- Models use mostly default hyperparameters (no tuning).
- The classifier learns writing patterns from this dataset; it does not verify facts and may not generalize to other sources without retraining.

## Possible Improvements

- 5-fold cross-validation for a more reliable ranking
- Hyperparameter tuning (for example `C` for Linear SVM) with `GridSearchCV`
- Larger `n_features` or word pairs via `ngram_range=(1, 2)`
- Testing on articles from outside this dataset

## Related

A TF-IDF version of this project is in `tfidf-fake-news-classifier`. TF-IDF scored about 0.6 points higher on average (best model 93.6% vs 93.1%), while hashing uses fixed memory and needs no stored vocabulary.
