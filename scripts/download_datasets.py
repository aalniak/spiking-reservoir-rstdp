#!/usr/bin/env python3
"""Pre-fetch / verify datasets.

* UEA datasets (BasicMotions, CharacterTrajectories, ...) are downloaded and
  cached via aeon. If aeon is unavailable the script reports it and continues.
* Synthetic datasets (Mackey-Glass, NARMA10, IQ jamming) are generated on the
  fly and only verified here (nothing to download).

    python scripts/download_datasets.py
    python scripts/download_datasets.py --uea BasicMotions CharacterTrajectories
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_dataset  # noqa: E402


def fetch_uea(names):
    try:
        from aeon.datasets import load_classification
    except Exception as e:  # pragma: no cover
        print(f"[download] aeon not available ({e}). UEA datasets will fall back / be skipped.")
        return
    for name in names:
        try:
            X, y = load_classification(name)
            print(f"[download] {name}: X={getattr(X, 'shape', 'list')} n={len(y)} "
                  f"classes={len(set(y))}  (cached)")
        except Exception as e:
            print(f"[download] FAILED {name}: {e}")


def verify_synthetic():
    for cfg in (
        {"name": "mackey_glass", "train_len": 300, "val_len": 100, "test_len": 100},
        {"name": "narma10", "train_len": 300, "val_len": 100, "test_len": 100},
        {"name": "synthetic_iq_jamming", "n_per_class": 40, "T": 128},
    ):
        b = load_dataset(cfg, seed=0)
        print(f"[verify] {b.name}: X_train={tuple(b.X_train.shape)} task={b.task_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uea", nargs="*", default=["BasicMotions", "CharacterTrajectories"])
    args = ap.parse_args()
    print("== UEA datasets (via aeon) ==")
    fetch_uea(args.uea)
    print("\n== synthetic datasets ==")
    verify_synthetic()
    print("\nDone.")


if __name__ == "__main__":
    main()
