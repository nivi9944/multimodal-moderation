import json

import joblib

import src.eval.plots as plots
import src.fusion.train_fusion as tf
from src.fusion.features import load_branch_scores, make_features


def test_features_logit_and_interaction(scores_dir):
    df = load_branch_scores(scores_dir)
    X, used = make_features(df, ["p_text", "p_image", "text_x_image", "p_clip"])
    assert used == ["p_text", "p_image", "text_x_image"]  # p_clip absent -> skipped
    assert X.shape == (len(df), 3)


def test_fusion_pipeline_end_to_end(scores_dir, tmp_path, monkeypatch):
    res, mod = tmp_path / "results", tmp_path / "models"
    res.mkdir()
    monkeypatch.setattr(tf, "SCORES_DIR", scores_dir)
    monkeypatch.setattr(tf, "RESULTS", res)
    monkeypatch.setattr(tf, "MODELS", mod)
    monkeypatch.setattr(plots, "RESULTS", res)
    tf.main(["--memes-root", str(tmp_path / "missing")])

    fe = json.loads((res / "fusion_eval.json").read_text())
    v = fe["variants"]
    assert {"text_only", "image_only", "late_fusion"} <= set(v)
    # the synthetic branches carry independent signal, so fusion must beat each one
    assert v["late_fusion"]["test"]["auroc"] > max(v["text_only"]["test"]["auroc"], v["image_only"]["test"]["auroc"])

    th = json.loads((res / "threshold.json").read_text())
    assert th["val_oof"]["at_t_clear"]["harmful_recall"] >= 0.9
    tri = json.loads((res / "triage.json").read_text())
    assert 0 < tri["auto_cleared_pct"] < 100
    assert any(k.startswith("fusion_") for k in json.loads((res / "calibration.json").read_text()))
    assert (res / "examples.md").exists() and (res / "fusion_vs_single.png").exists()
    bundle = joblib.load(mod / "fusion_lr.joblib")
    assert bundle["features"][0] == "p_text" and bundle["thresholds"]["t_clear"] <= bundle["thresholds"]["t_flag"]
