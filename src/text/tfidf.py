"""TF-IDF + logistic regression text classifier (kept in its own module so pickles load anywhere)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import SEED
from src.text.clean import clean_for_tfidf


class TfidfLogReg:
    """Word (1-2 gram) + char_wb (2-5 gram) TF-IDF features into class-weighted multinomial LR."""

    def __init__(self, max_word: int = 200_000, max_char: int = 200_000, C: float = 4.0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True)
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=max_char, sublinear_tf=True
        )
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)

    def _x(self, texts, fit: bool = False):
        clean = [clean_for_tfidf(t) for t in texts]
        if fit:
            return hstack([self.word.fit_transform(clean), self.char.fit_transform(clean)]).tocsr()
        return hstack([self.word.transform(clean), self.char.transform(clean)]).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._x(texts))
