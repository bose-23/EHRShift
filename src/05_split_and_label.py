import pandas as pd
from pathlib import Path
import importlib.util

BASE = Path(".")
OUT = Path("outputs")


def load_aggregated():
    p = OUT / "aggregated_patients.csv"
    df = pd.read_csv(p, dtype={"PATIENT": str}, parse_dates=["last_encounter"]) 
    df = df.set_index("PATIENT")
    return df


def make_labels():
    this_dir = Path(__file__).resolve().parents[0]
    spec = importlib.util.spec_from_file_location("label_mod", this_dir / "02_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    spec2 = importlib.util.spec_from_file_location("split_mod", this_dir / "01_split.py")
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    pl = mod2.compute_patient_last_encounter()
    cond = mod.load_conditions()
    meds = pd.read_csv(Path('csv') / 'medications.csv', dtype={"PATIENT": str}, parse_dates=["START"], engine='python', on_bad_lines='skip') 
    labels_base = mod.extract_diabetes_labels(cond, patient_last=pl)
    labels_aug = mod.augment_diabetes_labels(cond, meds, patient_last=pl)
    labels = labels_base.reindex(pl.index).fillna(0).astype(int) | labels_aug.reindex(pl.index).fillna(0).astype(int)
    return labels


def assign_splits(df, target_test_frac=0.2):
    patient_last = df["last_encounter"].dropna()
    cutoff = patient_last.quantile(1 - target_test_frac)
    train_ids = patient_last[patient_last < cutoff].index.tolist()
    test_ids = patient_last[patient_last >= cutoff].index.tolist()
    return cutoff, train_ids, test_ids


if __name__ == "__main__":
    df = load_aggregated()
    labels = make_labels()
    cutoff, train_ids, test_ids = assign_splits(df)
    print("Cutoff:", cutoff)
    print("Dataset 1:", len(train_ids), "Dataset 2:", len(test_ids))
    OUT.mkdir(exist_ok=True)
    pd.Series(train_ids, name="PATIENT").to_csv(OUT / "train_ids.csv", index=False)
    pd.Series(test_ids, name="PATIENT").to_csv(OUT / "test_ids.csv", index=False)
    labels.to_csv(OUT / "diabetes_labels.csv")
    print("Saved split IDs and labels to outputs/")
