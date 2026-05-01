import importlib.util
from pathlib import Path

def load_module_from_src(name):
    base = Path(__file__).resolve().parents[0]
    path = base / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

split_mod = load_module_from_src("01_split")
label_mod = load_module_from_src("02_label")

compute_patient_last_encounter = split_mod.compute_patient_last_encounter
choose_temporal_cutoff = split_mod.choose_temporal_cutoff
load_conditions = label_mod.load_conditions
extract_diabetes_labels = label_mod.extract_diabetes_labels


def main(target_test_frac=0.2, min_test_patients=200, min_positive_test=50):
    pl = compute_patient_last_encounter()
    cond = load_conditions()
    import pandas as pd
    meds = pd.read_csv(Path('csv') / 'medications.csv', dtype={"PATIENT": str}, parse_dates=["START"]) 
    labels = extract_diabetes_labels(cond, patient_last=pl)
    label_mod2 = load_module_from_src("02_label")
    aug = label_mod2.augment_diabetes_labels(cond, meds, patient_last=pl)
    labels_combined = labels.reindex(pl.index).fillna(0).astype(int) | aug.reindex(pl.index).fillna(0).astype(int)
    labels = labels_combined
    cutoff, d1, d2, frac, n_test, pos_test = choose_temporal_cutoff(pl, target_test_frac=target_test_frac, min_test_patients=min_test_patients, min_positive_test=min_positive_test, diabetes_labels=labels)
    print("Cutoff:", cutoff)
    print("Patients:", len(pl))
    print("Dataset 1:", len(d1), "Dataset 2:", len(d2), "fraction used:", frac)
    print("Total positive labels:", int(labels.sum()))
    if pos_test is not None:
        print("Positive labels in Dataset 2:", int(pos_test))


if __name__ == "__main__":
    main()
