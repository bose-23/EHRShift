import pandas as pd
from pathlib import Path
from typing import Set

CSV_DIR = Path("csv")
TARGET_KEYWORD = "Diabetes"

# ICD-10 diabetes codes usually fall under E10-E14.
DIABETES_CODE_PREFIXES = ["E10", "E11", "E12", "E13", "E14"]

# Common diabetes medication keywords.
ANTIDIABETIC_MEDS = [
    "metformin", "insulin", "glipizide", "glyburide", "glimepiride", "pioglitazone",
    "sitagliptin", "saxagliptin", "linagliptin", "empagliflozin", "canagliflozin",
    "dapagliflozin", "gliclazide", "alogliptin", "repaglinide"
]


def load_conditions(path: str = None):
    p = CSV_DIR / "conditions.csv" if path is None else Path(path)
    usecols = ["PATIENT", "START", "DESCRIPTION", "CODE"]
    cond = pd.read_csv(p, usecols=usecols, dtype={"PATIENT": str, "DESCRIPTION": str, "CODE": str}, parse_dates=["START"]) 
    cond["START"] = pd.to_datetime(cond["START"], errors="coerce")
    return cond


def extract_diabetes_labels(conditions: pd.DataFrame, patient_last: pd.Series = None, prevalent_at_last_encounter: bool = True):
    df = conditions.copy()
    df["DESCRIPTION"] = df["DESCRIPTION"].fillna("")
    mask = df["DESCRIPTION"].str.contains(TARGET_KEYWORD, case=False, na=False)
    diabetes = df[mask].copy()

    if diabetes.empty:
        return pd.Series(dtype=int)

    diabetes_event = diabetes.groupby("PATIENT", observed=True)["START"].min().rename("diabetes_first")

    labels = pd.Series(0, index=patient_last.index if patient_last is not None else diabetes_event.index, dtype=int)

    if patient_last is not None and prevalent_at_last_encounter:
        pl = patient_last.copy()
        try:
            if pl.dt.tz is not None:
                pl = pl.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        de = diabetes_event.copy()
        try:
            if de.dt.tz is not None:
                de = de.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        aligned = de.reindex(pl.index)
        pos = aligned <= pl
        labels[pos.fillna(False).index[pos.fillna(False)]] = 1
    else:
        labels.loc[diabetes_event.index] = 1

    return labels


def augment_diabetes_labels(conditions: pd.DataFrame, medications: pd.DataFrame, patient_last: pd.Series = None, prevalent_at_last_encounter: bool = True):
    """Build a broader diabetes label using condition text, ICD-like codes, and diabetes meds."""
    cond = conditions.copy()
    med = medications.copy()
    cond["DESCRIPTION"] = cond["DESCRIPTION"].fillna("")
    med["DESCRIPTION"] = med.get("DESCRIPTION", "").fillna("") if "DESCRIPTION" in med.columns else med.get("REASONDESCRIPTION", "")

    mask_desc = cond["DESCRIPTION"].str.contains(TARGET_KEYWORD, case=False, na=False)
    cond_desc = cond[mask_desc]

    mask_code = False
    if "CODE" in cond.columns:
        mask_code = cond["CODE"].fillna("").astype(str).apply(lambda x: any(x.startswith(pref) for pref in DIABETES_CODE_PREFIXES))
    cond_code = cond[mask_code]

    med_matches = pd.Series(False, index=med.index)
    for kw in ANTIDIABETIC_MEDS:
        med_matches = med_matches | med.get("DESCRIPTION", med.get("REASONDESCRIPTION", "")).str.contains(kw, case=False, na=False)
    meds_pos = med[med_matches]

    frames = []
    if not cond_desc.empty:
        tmp = cond_desc.copy()
        if "START" in tmp.columns:
            tmp["START"] = pd.to_datetime(tmp["START"], errors="coerce", utc=True)
        frames.append(tmp)
    if not cond_code.empty:
        tmp = cond_code.copy()
        if "START" in tmp.columns:
            tmp["START"] = pd.to_datetime(tmp["START"], errors="coerce", utc=True)
        frames.append(tmp)
    if not meds_pos.empty:
        tmp = meds_pos.copy()
        if "START" in tmp.columns:
            tmp["START"] = pd.to_datetime(tmp["START"], errors="coerce", utc=True)
        frames.append(tmp)
    if not frames:
        idx = patient_last.index if patient_last is not None else pd.Series(dtype=str)
        return pd.Series(0, index=idx, dtype=int)
    combined = pd.concat(frames, axis=0, ignore_index=False)
    if "START" in combined.columns:
        combined["START"] = combined["START"].dt.tz_convert('UTC')
        event = combined.groupby("PATIENT", observed=True)["START"].min().rename("first_event")
    else:
        event = pd.Series(1, index=combined["PATIENT"].unique()).rename("first_event")

    labels = pd.Series(0, index=patient_last.index if patient_last is not None else event.index, dtype=int)
    if patient_last is not None and prevalent_at_last_encounter and "START" in combined.columns:
        pl = patient_last.copy()
        try:
            if pl.dt.tz is not None:
                pl = pl.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        de = event.copy()
        try:
            if de.dt.tz is not None:
                de = de.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        aligned = de.reindex(pl.index)
        pos = aligned <= pl
        labels[pos.fillna(False).index[pos.fillna(False)]] = 1
    else:
        labels.loc[event.index] = 1

    return labels


if __name__ == "__main__":
    print("Loading conditions and building diabetes labels...")
    cond = load_conditions()
    import importlib.util
    from pathlib import Path
    spec_path = Path(__file__).resolve().parents[0] / "01_split.py"
    spec = importlib.util.spec_from_file_location("split_module", spec_path)
    split_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_module)
    pl = split_module.compute_patient_last_encounter()
    labels = extract_diabetes_labels(cond, patient_last=pl)
    print("Patients in label vector:", len(labels))
    print("Positive labels:", int(labels.sum()))
