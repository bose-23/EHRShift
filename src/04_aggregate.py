import pandas as pd
from pathlib import Path
import numpy as np
import json

CSV_DIR = Path("csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_patients(path: str = None):
    p = CSV_DIR / "patients.csv" if path is None else Path(path)
    usecols = ["Id", "BIRTHDATE", "GENDER", "RACE", "ZIP", "INCOME"]
    patients = pd.read_csv(p, usecols=usecols, dtype={"Id": str}, parse_dates=["BIRTHDATE"]) 
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients = patients.set_index("Id")
    return patients


def aggregate_basic_features(patient_last):
    patients = load_patients()
    df = patients.reindex(patient_last.index).copy()
    df["last_encounter"] = patient_last
    try:
        if hasattr(df["last_encounter"].dt, "tz") and df["last_encounter"].dt.tz is not None:
            df["last_encounter"] = df["last_encounter"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    try:
        if hasattr(df["BIRTHDATE"].dt, "tz") and df["BIRTHDATE"].dt.tz is not None:
            df["BIRTHDATE"] = df["BIRTHDATE"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    df["age_at_last"] = ((df["last_encounter"] - df["BIRTHDATE"]).dt.days / 365.25).astype(float)
    df["n_encounters"] = 0
    return df


def aggregate_encounters(patient_last):
    p = CSV_DIR / "encounters.csv"
    usecols = ["PATIENT", "START", "TOTAL_CLAIM_COST"]
    df = pd.read_csv(p, usecols=usecols, dtype={"PATIENT": str}, parse_dates=["START"], low_memory=False)
    df["START"] = pd.to_datetime(df["START"], errors="coerce")
    try:
        if df["START"].dt.tz is not None:
            df["START"] = df["START"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    df = df.dropna(subset=["START"]).copy()
    agg = {}
    for pid, g in df.groupby("PATIENT"):
        last = patient_last.get(pid)
        try:
            if hasattr(last, 'tz') and last.tz is not None:
                last = last.tz_convert('UTC').tz_localize(None)
        except Exception:
            pass
        if pd.isna(last):
            continue
        gsub = g[g["START"] <= last]
        if gsub.empty:
            continue
        n = len(gsub)
        cost_mean = pd.to_numeric(gsub["TOTAL_CLAIM_COST"], errors="coerce").mean()
        agg[pid] = {"n_encounters": n, "enc_cost_mean": cost_mean}
    rows = []
    for pid, v in agg.items():
        v2 = {f"enc_{k}": val for k, val in v.items()}
        rows.append({"Id": pid, **v2})
    if rows:
        df = pd.DataFrame(rows).set_index("Id")
    else:
        df = pd.DataFrame(index=pd.Index([], name="Id"))
    return df


def aggregate_medications(patient_last):
    p = CSV_DIR / "medications.csv"
    usecols = ["PATIENT", "START", "DESCRIPTION", "REASONDESCRIPTION"]
    df = pd.read_csv(p, usecols=[c for c in usecols if c in pd.read_csv(p, nrows=0).columns], dtype={"PATIENT": str}, parse_dates=["START"], low_memory=True)
    df["START"] = pd.to_datetime(df["START"], errors="coerce")
    try:
        if df["START"].dt.tz is not None:
            df["START"] = df["START"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    agg = {}
    for pid, g in df.groupby("PATIENT"):
        last = patient_last.get(pid)
        try:
            if hasattr(last, 'tz') and last.tz is not None:
                last = last.tz_convert('UTC').tz_localize(None)
        except Exception:
            pass
        if pd.isna(last):
            continue
        gsub = g[g["START"] <= last]
        if gsub.empty:
            continue
        n = len(gsub)
        unique_meds = gsub["DESCRIPTION"].nunique() if "DESCRIPTION" in gsub.columns else 0
        agg[pid] = {"med_count": n, "med_unique": unique_meds}
    rows = []
    for pid, v in agg.items():
        rows.append({"Id": pid, **v})
    if rows:
        return pd.DataFrame(rows).set_index("Id")
    return pd.DataFrame(index=pd.Index([], name="Id"))


def aggregate_conditions(patient_last):
    p = CSV_DIR / "conditions.csv"
    usecols = ["PATIENT", "START", "DESCRIPTION", "CODE"]
    df = pd.read_csv(p, usecols=[c for c in usecols if c in pd.read_csv(p, nrows=0).columns], dtype={"PATIENT": str}, parse_dates=["START"], low_memory=True)
    df["START"] = pd.to_datetime(df["START"], errors="coerce")
    try:
        if df["START"].dt.tz is not None:
            df["START"] = df["START"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    agg = {}
    for pid, g in df.groupby("PATIENT"):
        last = patient_last.get(pid)
        if pd.isna(last):
            continue
        gsub = g[g["START"] <= last]
        if gsub.empty:
            continue
        n = len(gsub)
        unique_codes = gsub["CODE"].nunique() if "CODE" in gsub.columns else 0
        agg[pid] = {"cond_count": n, "cond_unique": unique_codes}
    rows = []
    for pid, v in agg.items():
        rows.append({"Id": pid, **v})
    if rows:
        return pd.DataFrame(rows).set_index("Id")
    return pd.DataFrame(index=pd.Index([], name="Id"))


def aggregate_observations(patient_last, features_of_interest=None):
    if features_of_interest is None:
        features_of_interest = ["Body Height", "Body Weight", "BMI", "Glucose", "Hemoglobin A1c", "HbA1c"]
    p = CSV_DIR / "observations.csv"
    usecols = ["PATIENT", "DATE", "DESCRIPTION", "VALUE"]
    it = pd.read_csv(p, usecols=usecols, dtype={"PATIENT": str, "DESCRIPTION": str}, chunksize=500000)
    acc = {}
    for chunk in it:
        chunk["DATE"] = pd.to_datetime(chunk["DATE"], errors="coerce")
        try:
            if chunk["DATE"].dt.tz is not None:
                chunk["DATE"] = chunk["DATE"].dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        chunk = chunk.dropna(subset=["DATE"]).copy()
        for kw in features_of_interest:
            sub = chunk[chunk["DESCRIPTION"].str.contains(kw, case=False, na=False)]
            if sub.empty:
                continue
            for pid, g in sub.groupby("PATIENT"):
                last = patient_last.get(pid)
                if pd.isna(last):
                    continue
                g = g[g["DATE"] <= last]
                if g.empty:
                    continue
                vals = pd.to_numeric(g["VALUE"], errors="coerce").dropna()
                if vals.empty:
                    continue
                key_mean = f"{kw}_mean"
                key_last = f"{kw}_last"
                if pid not in acc:
                    acc[pid] = {}
                acc[pid][key_mean] = vals.mean()
                acc[pid][key_last] = vals.iloc[-1]
    rows = []
    for pid, v in acc.items():
        rows.append({"Id": pid, **v})
    if rows:
        df = pd.DataFrame(rows).set_index("Id")
    else:
        df = pd.DataFrame(index=pd.Index([], name="Id"))
    return df


def build_patient_table():
    patient_last = None
    import importlib.util
    from pathlib import Path
    spec_path = Path(__file__).resolve().parents[0] / "01_split.py"
    spec = importlib.util.spec_from_file_location("split_module", spec_path)
    split_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_module)
    patient_last = split_module.compute_patient_last_encounter()
    try:
        if hasattr(patient_last.dt, 'tz') and patient_last.dt.tz is not None:
            patient_last = patient_last.dt.tz_convert('UTC').dt.tz_localize(None)
    except Exception:
        pass

    base = aggregate_basic_features(patient_last)
    enc = aggregate_encounters(patient_last)
    obs = aggregate_observations(patient_last)
    meds = aggregate_medications(patient_last)
    conds = aggregate_conditions(patient_last)

    df = base.join(enc, how="left").join(obs, how="left").join(meds, how="left").join(conds, how="left")
    # Fill BMI from height and weight when it is missing.
    if "Body Weight_last" in df.columns and "Body Height_last" in df.columns:
        try:
            wt = pd.to_numeric(df["Body Weight_last"], errors="coerce")
            ht_cm = pd.to_numeric(df["Body Height_last"], errors="coerce")
            ht_m = ht_cm / 100.0
            bmi_calc = wt / (ht_m * ht_m)
            df["BMI_last"] = df["BMI_last"].fillna(bmi_calc)
        except Exception:
            pass
    out = OUT_DIR / "aggregated_patients.csv"
    df.to_csv(out)
    print("Saved patient table to", out)
    return df


if __name__ == "__main__":
    build_patient_table()
