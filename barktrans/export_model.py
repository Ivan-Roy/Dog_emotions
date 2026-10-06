"""
export_model.py — convert a trained .joblib model into web/model.json so the
browser can run it with no Python. Also writes web/reference.json: the exact
feature vector and prediction for one sample clip, so the JS app can prove its
feature extraction matches Python.

Usage:
    python -m barktrans.export_model \
        --model models/bark_model.joblib \
        --sample path/to/any_bark.wav \
        --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from .features import (extract_features, TARGET_SR, N_MFCC, N_MELS,
                       FRAME_LEN, HOP_LEN, FMIN, FMAX)
from .predict import predict_file, translate


def _tree_to_dict(tree) -> dict:
    """Serialize one sklearn decision tree to plain arrays."""
    t = tree.tree_
    # value is (n_nodes, 1, n_classes) -> per-node class counts; normalize to proba
    value = t.value.reshape(t.value.shape[0], -1)
    proba = value / value.sum(axis=1, keepdims=True)
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": proba.tolist(),
    }


def export(model_path: str, sample_wav: str, out_dir: str) -> None:
    bundle = joblib.load(model_path)
    os.makedirs(out_dir, exist_ok=True)

    model_json = {
        "label_cols": bundle["label_cols"],
        "feature_names": bundle["feature_names"],
        "params": {
            "target_sr": TARGET_SR, "n_mfcc": N_MFCC, "n_mels": N_MELS,
            "frame_len": FRAME_LEN, "hop_len": HOP_LEN,
            "fmin": FMIN, "fmax": FMAX,
        },
        "labels": {},
    }

    for col in bundle["label_cols"]:
        clf, le = bundle["models"][col], bundle["encoders"][col]
        model_json["labels"][col] = {
            "classes": le.classes_.tolist(),
            "trees": [_tree_to_dict(est) for est in clf.estimators_],
        }

    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(model_json, f)

    # reference vector + prediction for the JS self-test
    vec, names = extract_features(sample_wav)
    pred = predict_file(bundle, sample_wav)
    reference = {
        "feature_names": names,
        "features": [round(float(v), 6) for v in vec],
        "prediction": {k: v["label"] for k, v in pred.items()},
        "translation": translate(pred)["headline"],
    }
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(reference, f, indent=2)

    n_trees = sum(len(model_json["labels"][c]["trees"]) for c in bundle["label_cols"])
    print(f"Exported model.json ({n_trees} trees) and reference.json -> {out_dir}/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export model for the web app.")
    ap.add_argument("--model", default="models/bark_model.joblib")
    ap.add_argument("--sample", required=True, help="any wav, for the self-test")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args(argv)
    export(args.model, args.sample, args.out_dir)


if __name__ == "__main__":
    main()
