"""Late fusion on Hateful Memes, with threshold tuning, triage and calibration. CPU, seconds.

Usage:
    python -m src.fusion.train_fusion [--memes-root PATH]

Input: data/processed/meme_scores_*.csv (from the GPU notebook).
For every feature set in features.VARIANTS:
    * standardised logistic regression trained on fusion_train (5-fold out-of-fold scores for tuning)
    * evaluated on test: AUROC, average precision, macro-F1 at the OOF-best-F1 threshold
For the headline variant (best out-of-fold AUROC):
    * t_clear (Harmful recall >= 90% on OOF) and t_flag (precision >= 0.8) -> results/threshold.json
    * triage metric on test -> results/triage.json
    * failure examples -> results/examples.md
Temperature scaling ECE for the headline and deployed variants -> results/calibration.json (keys fusion_*).
Model bundle for the API (deployed variant late_fusion, with its own thresholds): models/fusion_lr.joblib
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, MODELS, RESULTS, SEED, TARGET_RECALL, set_seed, write_json
from src.eval.metrics import binary_report
from src.eval.triage import triage_report
from src.fusion.calibrate import calibration_report, fit_temperature_binary
from src.fusion.features import DEPLOYED, VARIANTS, load_branch_scores, make_features
from src.fusion.threshold import decide, fit_thresholds, threshold_for_recall, threshold_report

SCORES_DIR = DATA_PROCESSED  # module-level so tests can point it at fixtures


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))


def best_f1_threshold(y, p) -> float:
    """Threshold maximising macro-F1 over the 2nd to 98th percentiles of p (97-point grid)."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, p >= t, average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def fit_variant(df: pd.DataFrame, cols: list[str]):
    """Fit one feature set on fusion_train and score test. Returns None if none of its columns exist."""
    tr, te = df[df.split == "fusion_train"], df[df.split == "test"]
    Xtr, used = make_features(tr, cols)
    if not used:
        return None
    Xte, _ = make_features(te, cols)
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = cross_val_predict(make_model(), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    model = make_model().fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    t = best_f1_threshold(ytr, p_oof)
    return {
        "features": used,
        "model": model,
        "p_oof": p_oof,
        "p_test": p_te,
        "y_oof": ytr,
        "y_test": yte,
        "report": {"features": used, "oof": binary_report(ytr, p_oof, t), "test": binary_report(yte, p_te, t)},
    }


def write_examples(df_test: pd.DataFrame, p: np.ndarray, th, captions: dict | None, model: str, k: int = 5) -> None:
    """Write the k most confident misses and false alarms on test to results/examples.md."""
    d = df_test.copy()
    d["p_harmful"] = p
    d["decision"] = np.array(["Safe", "Neutral", "Harmful"])[decide(p, th)]
    fn = d[(d.label == 1) & (d.decision == "Safe")].sort_values("p_harmful").head(k)
    fp = d[(d.label == 0) & (d.decision == "Harmful")].sort_values("p_harmful", ascending=False).head(k)
    cols = [c for c in ("p_text", "p_text_jigsaw", "p_image", "p_clip", "weapon_max_conf", "p_harmful") if c in d]
    lines = [
        f"# Failure examples (Hateful Memes test split, {model} model)",
        "",
        f"Thresholds: t_clear = {th.t_clear:.4f}, t_flag = {th.t_flag:.4f}. Captions are the dataset's "
        "overlaid text; images are not reproduced here.",
        "",
    ]
    for title, sub in (
        ("Missed harmful memes (auto-cleared as Safe)", fn),
        ("False alarms (auto-flagged Harmful)", fp),
    ):
        lines += [f"## {title}", "", "| id | caption | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for _, r in sub.iterrows():
            cap = (captions or {}).get(r["id"], "")
            cap = str(cap).replace("|", "/")[:90]
            lines.append(f"| {r['id']} | {cap} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        lines += ["", "Why it fails: _fill in after reading the examples (see HANDOVER step 6)._", ""]
    (RESULTS / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memes-root")
    args = ap.parse_args(argv)
    set_seed()
    df = load_branch_scores(SCORES_DIR)
    fits, seen = {}, set()
    for name, cols in VARIANTS.items():
        f = fit_variant(df, cols)
        if f is None or tuple(f["features"]) in seen:
            continue  # no usable columns, or missing branch files reduced it to a variant already fitted
        seen.add(tuple(f["features"]))
        fits[name] = f

    # thresholds and temperature for the API bundle come from the deployed variant
    dep = fits[DEPLOYED]
    th = fit_thresholds(dep["y_oof"], dep["p_oof"], TARGET_RECALL)
    t_temp = fit_temperature_binary(dep["y_oof"], dep["p_oof"])

    # Headline model: best out-of-fold AUROC on fusion_train, never chosen on test.
    head_name = max(fits, key=lambda k: fits[k]["report"]["oof"].get("auroc", 0))
    head = fits[head_name]
    th_head = fit_thresholds(head["y_oof"], head["p_oof"], TARGET_RECALL)

    single = {k: fits[k]["report"]["test"].get("auroc") for k in ("text_only", "image_only") if k in fits}
    best_single = max(single, key=single.get) if single else None
    write_json(
        RESULTS / "fusion_eval.json",
        {
            "dataset": "Hateful Memes",
            "eval_split_note": "test = labelled split named in data_summary.json (hateful_memes.test_file)",
            "protocol": "branches trained on branch_train; fusion LR trained on fusion_train (held out); thresholds from 5-fold OOF on fusion_train; metrics on test",
            "variants": {k: v["report"] for k, v in fits.items()},
            "headline": head_name,
            "headline_rule": "highest out-of-fold AUROC on fusion_train",
            "deployed_api": DEPLOYED,
            "best_single_branch": best_single,
            "fusion_minus_best_single_auroc": round(dep["report"]["test"]["auroc"] - single[best_single], 4)
            if best_single
            else None,
            "headline_minus_text_only_auroc": round(head["report"]["test"]["auroc"] - single["text_only"], 4)
            if "text_only" in single
            else None,
        },
        merge=False,
    )

    per_variant = {}
    for k, v in fits.items():
        t_k = fit_thresholds(v["y_oof"], v["p_oof"], TARGET_RECALL)
        tr_k = triage_report(v["y_test"], v["p_test"], t_k.t_clear)
        per_variant[k] = {
            "t_clear": round(t_k.t_clear, 6),
            "oof_harmful_recall": threshold_report(v["y_oof"], v["p_oof"], t_k)["at_t_clear"]["harmful_recall"],
            "test_harmful_recall": tr_k["harmful_recall"],
            "test_harmful_precision": threshold_report(v["y_test"], v["p_test"], t_k)["at_t_clear"][
                "harmful_precision"
            ],
            "test_auto_cleared_pct": tr_k["auto_cleared_pct"],
        }
    rep = threshold_report(head["y_test"], head["p_test"], th_head)
    rep["variant"] = head_name
    rep["val_oof"] = threshold_report(head["y_oof"], head["p_oof"], th_head)
    rep["per_variant"] = per_variant
    rep["note"] = (
        f"Thresholds chosen on fusion_train out-of-fold scores (target Harmful recall {TARGET_RECALL}); applied unchanged to test."
    )
    write_json(RESULTS / "threshold.json", rep, merge=False)
    tri = triage_report(head["y_test"], head["p_test"], th_head.t_clear)
    tri["variant"] = head_name
    # Reference only (uses test labels, never used for any decision): the share that could be cleared if the
    # threshold were set on test itself to hit exactly 90% recall. It sizes the cost of the val-to-test shift.
    t_oracle = threshold_for_recall(head["y_test"], head["p_test"], TARGET_RECALL)
    tri["reference_test_oracle_at_target_recall"] = {
        "auto_cleared_pct": triage_report(head["y_test"], head["p_test"], t_oracle)["auto_cleared_pct"],
        "note": "threshold picked on test labels; for context only, not a result of the method",
    }
    tri["deployed_api"] = {DEPLOYED: per_variant[DEPLOYED]}
    tri["definition"] = (
        "auto_cleared_pct = share of test items with p_harmful < t_clear, where t_clear keeps "
        "Harmful recall >= 90% on validation (out-of-fold). These items skip manual review."
    )
    write_json(RESULTS / "triage.json", tri, merge=False)
    write_json(
        RESULTS / "calibration.json",
        {
            f"fusion_{head_name}": calibration_report(head["y_oof"], head["p_oof"], head["y_test"], head["p_test"]),
            f"fusion_{DEPLOYED}": calibration_report(dep["y_oof"], dep["p_oof"], dep["y_test"], dep["p_test"]),
        },
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": dep["model"], "features": dep["features"], "thresholds": th.to_dict(), "temperature": t_temp},
        MODELS / "fusion_lr.joblib",
    )

    captions = None
    try:
        from src.image.memes import load_memes

        s, _ = load_memes(args.memes_root)
        captions = dict(zip(s["test"]["id"].astype(int), s["test"]["text"]))
    except FileNotFoundError:
        pass
    te = df[df.split == "test"].reset_index(drop=True)
    write_examples(te, head["p_test"], th_head, captions, head_name)

    from src.eval.plots import plot_fusion

    plot_fusion({k: (v["y_test"], v["p_test"]) for k, v in fits.items()}, out_dir=RESULTS)
    for k, v in fits.items():
        print(f"{k:>22}: test AUROC {v['report']['test'].get('auroc')}  macro-F1 {v['report']['test']['macro_f1']}")
    print("headline:", head_name, "triage:", tri["auto_cleared_pct"], "% auto-cleared at recall", tri["harmful_recall"])


if __name__ == "__main__":
    main()
