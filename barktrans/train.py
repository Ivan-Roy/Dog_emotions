"""
train.py — train a bark emotion classifier from a labeled dataset.

Designed for the Barkopedia Dog Emotion dataset, but flexible: point it at a
folder of .wav files and a CSV of labels. The CSV needs one column naming the
audio file and one or more label columns (default: arousal, valence).

Example:
    python -m barktrans.train \
        --audio-dir data/husky_train \
        --labels data/husky_train_labels.csv \
        --audio-col audio_id \
        --label-cols arousal,valence \
        --out models/bark_model.joblib

It prints cross-validated accuracy per label so you see, honestly, how well
each one is actually learnable from the data.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from .features import extract_features


def _resolve_audio_path(audio_dir: str, name: str) -> str | None:
    """Find the wav for a label row, tolerating missing/extra .wav extension."""
    for candidate in (name, f"{name}.wav", name.replace(".wav", "") + ".wav"):
        p = os.path.join(audio_dir, candidate)
        if os.path.isfile(p):
            return p
    return None


def build_dataset(audio_dir: str, labels_csv: str, audio_col: str,
                  label_cols: list[str]) -> tuple[np.ndarray, dict, list[str], list[str]]:
    df = pd.read_csv(labels_csv)

    missing = [c for c in [audio_col, *label_cols] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV is missing columns {missing}. Found: {list(df.columns)}"
        )

    X, kept_rows, feature_names = [], [], None
    skipped = 0
    for _, row in df.iterrows():
        path = _resolve_audio_path(audio_dir, str(row[audio_col]))
        if path is None:
            skipped += 1
            continue
        try:
            vec, names = extract_features(path)
        except Exception as e:  # noqa: BLE001 — skip unreadable clips, keep going
            print(f"  ! skipping {path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        X.append(vec)
        kept_rows.append(row)
        feature_names = names

    if not X:
        raise SystemExit("No usable audio found. Check --audio-dir / --audio-col.")

    print(f"Loaded {len(X)} clips ({skipped} skipped).")
    X = np.vstack(X)
    y = {col: np.array([r[col] for r in kept_rows]) for col in label_cols}
    return X, y, feature_names, label_cols


def train(X: np.ndarray, y: dict, label_cols: list[str]) -> dict:
    """Train one RandomForest per label. Returns a bundle dict to save."""
    models, encoders = {}, {}

    for col in label_cols:
        le = LabelEncoder()
        y_enc = le.fit_transform(y[col].astype(str))
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )

        # honest accuracy: 5-fold cross-validation (falls back if a class is tiny)
        n_splits = min(5, np.bincount(y_enc).min()) if len(np.unique(y_enc)) > 1 else 0
        if n_splits >= 2:
            scores = cross_val_score(clf, X, y_enc, cv=n_splits)
            print(f"  [{col}] cross-val accuracy: "
                  f"{scores.mean():.1%}  (+/- {scores.std():.1%})  "
                  f"across classes {list(le.classes_)}")
        else:
            print(f"  [{col}] too few samples per class for cross-val.")

        clf.fit(X, y_enc)
        models[col] = clf
        encoders[col] = le

    return {"models": models, "encoders": encoders, "label_cols": label_cols}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a bark emotion classifier.")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--labels", required=True, help="CSV of labels")
    ap.add_argument("--audio-col", default="audio_id")
    ap.add_argument("--label-cols", default="arousal,valence",
                    help="comma-separated label columns to train on")
    ap.add_argument("--out", default="models/bark_model.joblib")
    args = ap.parse_args(argv)

    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]

    print("Extracting features...")
    X, y, feature_names, label_cols = build_dataset(
        args.audio_dir, args.labels, args.audio_col, label_cols
    )

    print("Training...")
    bundle = train(X, y, label_cols)
    bundle["feature_names"] = feature_names

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved model -> {args.out}")


if __name__ == "__main__":
    main()
