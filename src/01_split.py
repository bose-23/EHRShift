import pandas as pd
import numpy as np
from pathlib import Path

CSV_DIR = Path("csv")


def compute_patient_last_encounter(encounters_csv: str = None):
    path = CSV_DIR / "encounters.csv" if encounters_csv is None else Path(encounters_csv)
    usecols = ["PATIENT", "START"]
    enc = pd.read_csv(path, usecols=usecols, dtype={"PATIENT": str}, parse_dates=["START"])
    enc["START"] = pd.to_datetime(enc["START"], errors="coerce")
    enc = enc.dropna(subset=["START"]).copy()
    patient_last = enc.groupby("PATIENT", observed=True)["START"].max().rename("last_encounter")
    return patient_last


def choose_temporal_cutoff(patient_last: pd.Series, target_test_frac: float = 0.2, min_test_patients: int = 200, min_positive_test: int = 50, diabetes_labels: pd.Series = None):
    q = 1 - target_test_frac
    cutoff = patient_last.quantile(q)
    dataset2_ids = patient_last[patient_last >= cutoff].index
    dataset1_ids = patient_last[patient_last < cutoff].index

    def checks(d2_ids):
        n_test = len(d2_ids)
        pos_test = int(diabetes_labels.loc[d2_ids].sum()) if (diabetes_labels is not None and len(d2_ids) > 0) else None
        return n_test, pos_test

    if diabetes_labels is not None:
        n_test, pos_test = checks(dataset2_ids)
        frac = target_test_frac
        while (n_test < min_test_patients or (pos_test is not None and pos_test < min_positive_test)) and frac <= 0.4:
            frac += 0.05
            q = 1 - frac
            cutoff = patient_last.quantile(q)
            dataset2_ids = patient_last[patient_last >= cutoff].index
            dataset1_ids = patient_last[patient_last < cutoff].index
            n_test, pos_test = checks(dataset2_ids)
        return cutoff, dataset1_ids, dataset2_ids, frac, n_test, pos_test

    return cutoff, dataset1_ids, dataset2_ids, target_test_frac, len(dataset2_ids), None


if __name__ == "__main__":
    print("Finding each patient's most recent encounter...")
    pl = compute_patient_last_encounter()
    cutoff, d1, d2, frac, n_test, pos_test = choose_temporal_cutoff(pl)
    print("Cutoff:", cutoff)
    print("Patients:", len(pl))
    print("Dataset 1:", len(d1), "Dataset 2:", len(d2), "fraction used:", frac)
