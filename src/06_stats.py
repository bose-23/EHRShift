import pandas as pd
from pathlib import Path

OUT = Path("outputs")


def main():
    df = pd.read_csv(OUT / "aggregated_patients.csv", dtype={"PATIENT": str}, parse_dates=["last_encounter"]).set_index("PATIENT")
    labels = pd.read_csv(OUT / "diabetes_labels.csv", index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    train = pd.read_csv(OUT / "train_ids.csv")["PATIENT"].tolist()
    test = pd.read_csv(OUT / "test_ids.csv")["PATIENT"].tolist()
    total = len(df)
    n_pos = int(labels.sum())
    train_pos = int(labels.reindex(train).sum())
    test_pos = int(labels.reindex(test).sum())
    print(f"Total patients: {total}, positive labels: {n_pos}")
    print(f"Dataset 1: n={len(train)}, positive={train_pos}")
    print(f"Dataset 2: n={len(test)}, positive={test_pos}")


if __name__ == "__main__":
    main()
