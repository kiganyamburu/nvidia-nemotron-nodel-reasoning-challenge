from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("train.csv")
texts = df["prompt"].astype(str).tolist()
vectorizer = TfidfVectorizer(
    lowercase=True, ngram_range=(1, 2), max_features=8000, token_pattern=r"(?u)\b\w+\b"
)
X = vectorizer.fit_transform(texts)
print("vocab size", len(vectorizer.vocabulary_))
print("sample vocab keys:", list(list(vectorizer.vocabulary_.keys())[:20]))
