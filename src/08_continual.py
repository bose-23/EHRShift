import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

OUT = Path("outputs")
MODEL_DIR = OUT / "models"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(OUT / "aggregated_patients.csv", dtype={"PATIENT": str}, parse_dates=["last_encounter"]).set_index("PATIENT")
    labels = pd.read_csv(OUT / "diabetes_labels.csv", index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    train_ids = pd.read_csv(OUT / "train_ids.csv")["PATIENT"].tolist()
    test_ids = pd.read_csv(OUT / "test_ids.csv")["PATIENT"].tolist()
    return df, labels, train_ids, test_ids


def transform(pre, X):
    return pre.transform(X)


def evaluate_model(model, X_t, y_true):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_t)[:, 1]
    else:
        # decision function fallback
        try:
            probs = model.decision_function(X_t)
            # scale to 0-1
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
        except Exception:
            probs = model.predict(X_t)
    fpr, tpr, _ = roc_curve(y_true, probs) if len(np.unique(y_true)) > 1 else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None else np.nan
    auprc = average_precision_score(y_true, probs)
    preds = model.predict(X_t)
    cm = confusion_matrix(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, preds)
    return dict(
        roc_auc=roc_auc,
        auprc=auprc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        tpr=tpr,
        preds=preds,
        probs=probs,
        cm=cm,
    )


def plot_roc(res_dict, name):
    plt.figure()
    for label, res in res_dict.items():
        fpr, tpr = res.get('fpr'), res.get('tpr')
        if fpr is None:
            continue
        plt.plot(fpr, tpr, label=f"{label} (AUC={res['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curves - {name}')
    plt.legend()
    out = FIG_DIR / f'roc_{name}.png'
    plt.savefig(out)
    plt.close()


def plot_confusion(cm, labels, name):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    out = FIG_DIR / f'cm_{name}.png'
    plt.savefig(out)
    plt.close()


def main():
    df, labels, train_ids, test_ids = load_data()

    pre = joblib.load(MODEL_DIR / 'preprocessor.joblib')
    dt1 = joblib.load(MODEL_DIR / 'dt_stage1.joblib')
    mlp1 = joblib.load(MODEL_DIR / 'mlp_stage1.joblib')

    d2_ids = test_ids
    if len(d2_ids) < 5:
        print('Dataset 2 is too small for a fine-tuning split. Skipping this step.')
        return
    split_idx = int(len(d2_ids) * 0.7)
    d2_train_ids = d2_ids[:split_idx]
    d2_test_ids = d2_ids[split_idx:]

    X_d2train = df.loc[d2_train_ids]
    y_d2train = labels.reindex(d2_train_ids).fillna(0).astype(int)
    X_d2test = df.loc[d2_test_ids]
    y_d2test = labels.reindex(d2_test_ids).fillna(0).astype(int)

    X_d2train_t = pre.transform(X_d2train)
    X_d2test_t = pre.transform(X_d2test)

    try:
        mlp1.warm_start = True
    except Exception:
        pass
    print('Fine-tuning the MLP on Dataset 2...')
    mlp1.max_iter = mlp1.max_iter + 100
    mlp1.fit(X_d2train_t, y_d2train)
    joblib.dump(mlp1, MODEL_DIR / 'mlp_finetuned.joblib')

    df_all_train_ids = list(train_ids) + list(d2_train_ids)
    X_comb = df.loc[df_all_train_ids]
    y_comb = labels.reindex(df_all_train_ids).fillna(0).astype(int)
    X_comb_t = pre.transform(X_comb)

    dt2 = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt2.fit(X_comb_t, y_comb)
    joblib.dump(dt2, MODEL_DIR / 'dt_retrained.joblib')

    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    X_d1 = df.loc[train_ids]
    y_d1 = labels.reindex(train_ids).fillna(0).astype(int)
    X_d1_t = pre.transform(X_d1)
    svc.fit(X_d1_t, y_d1)
    joblib.dump(svc, MODEL_DIR / 'svc_stage1.joblib')

    X_test = df.loc[test_ids]
    y_test = labels.reindex(test_ids).fillna(0).astype(int)
    X_test_t = pre.transform(X_test)

    res = {}
    res['mlp_finetuned'] = evaluate_model(mlp1, X_test_t, y_test)
    res['dt_retrained'] = evaluate_model(dt2, X_test_t, y_test)
    res['svc_stage1'] = evaluate_model(svc, X_test_t, y_test)

    joblib.dump(res, MODEL_DIR / 'results_stage2.joblib')
    stage2_rows = []
    for name, metrics in res.items():
        stage2_rows.append({
            'stage': 'stage2',
            'model': name.replace('_', ' ').title(),
            'evaluation_split': 'Dataset 2 test',
            'roc_auc': metrics.get('roc_auc'),
            'auprc': metrics.get('auprc'),
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1': metrics.get('f1'),
        })
    stage2_df = pd.DataFrame(stage2_rows)
    stage2_df.to_csv(REPORT_DIR / 'models_metrics_stage2.csv', index=False)

    stage1_path = REPORT_DIR / 'models_metrics_stage1.csv'
    if stage1_path.exists():
        stage1_df = pd.read_csv(stage1_path)
        pd.concat([stage1_df, stage2_df], ignore_index=True).to_csv(REPORT_DIR / 'models_metrics.csv', index=False)
    else:
        stage2_df.to_csv(REPORT_DIR / 'models_metrics.csv', index=False)
    print('Saved stage 2 results to', MODEL_DIR / 'results_stage2.joblib')

    plot_roc(res, 'dataset2_test')

    for name, info in res.items():
        cm = info['cm']
        plot_confusion(cm, ['neg','pos'], name)


if __name__ == '__main__':
    main()
