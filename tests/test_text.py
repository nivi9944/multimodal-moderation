import numpy as np
import pandas as pd

from src.text.baselines import vader_predict
from src.text.clean import clean_for_tfidf, normalise
from src.text.data import map_civil, map_jigsaw, map_youtoxic


def test_normalise_strips_noise_keeps_case():
    s = normalise("You are GREAT!!! see http://x.com @bob 10.0.0.1 (talk) soooooo &amp; more")
    assert "http" not in s and "@bob" not in s and "10.0.0.1" not in s and "(talk)" not in s
    assert "GREAT!!!" in s and "sooo " in s and "&" in s


def test_clean_for_tfidf_lowercases_and_keeps_masked_profanity():
    assert clean_for_tfidf("F*CK   this #$%") == "f*ck this"
    assert clean_for_tfidf(None) == ""


def _jig(rows):
    cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    return pd.DataFrame(rows, columns=cols)


def test_jigsaw_mild_mapping():
    df = _jig(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
        ]
    )
    assert map_jigsaw(df, "mild").tolist() == [0, 1, 1, 2, 2, 2]


def test_jigsaw_spec_mapping():
    df = _jig([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    assert map_jigsaw(df, "spec").tolist() == [0, 2, 1, 2]


def test_youtoxic_mapping():
    df = pd.DataFrame(
        {
            "IsToxic": [False, True, True],
            "IsObscene": [False, True, False],
            "IsHatespeech": [False, False, True],
            "Text": ["a", "b", "c"],
        }
    )
    assert map_youtoxic(df).tolist() == [0, 1, 2]


def test_civil_mapping():
    df = pd.DataFrame(
        {
            "toxicity": [0.1, 0.6, 0.7],
            "severe_toxicity": [0, 0, 0],
            "threat": [0, 0, 0.6],
            "insult": [0, 0.2, 0],
            "identity_attack": [0, 0, 0],
        }
    )
    assert map_civil(df).tolist() == [0, 1, 2]


def test_vader_predict_orders_thresholds():
    c = np.array([-0.9, -0.3, 0.5])
    assert vader_predict(c, t_harm=-0.5, t_neutral=0.0).tolist() == [2, 1, 0]
