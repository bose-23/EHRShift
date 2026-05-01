import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

OUT = Path("outputs")
MODEL_DIR = OUT / "models"
REPORT_DIR = OUT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(OUT / "aggregated_patients.csv", dtype={"PATIENT": str}, parse_dates=["last_encounter"]).set_index("PATIENT")
    labels = pd.read_csv(OUT / "diabetes_labels.csv", index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    train_ids = pd.read_csv(OUT / "train_ids.csv")["PATIENT"].tolist()
    test_ids = pd.read_csv(OUT / "test_ids.csv")["PATIENT"].tolist()
    return df, labels, train_ids, test_ids


def prepare_features(df):
    drop_cols = {"last_encounter", "BIRTHDATE", "ZIP", "n_encounters"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_df.columns if c not in num_cols]
    num_p = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat_p = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_p, num_cols), ('cat', cat_p, cat_cols)])
    return pre, num_cols, cat_cols


def train_and_eval():
    df, labels, train_ids, test_ids = load_data()
    df_train = df.loc[train_ids].copy()
    y_train = labels.reindex(df_train.index).fillna(0).astype(int)
    X_train_full = df_train
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.2, stratify=y_train, random_state=42)

    pre, num_cols, cat_cols = prepare_features(df)
    pre.fit(X_tr)
    joblib.dump(pre, MODEL_DIR / 'preprocessor.joblib')

    X_tr_t = pre.transform(X_tr)
    X_val_t = pre.transform(X_val)

    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr_t, y_tr)

    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt_params = {'max_depth':[None, 5, 10, 20], 'min_samples_leaf':[1,5,10]}
    dt_search = GridSearchCV(dt, dt_params, scoring='average_precision', cv=3, n_jobs=-1)
    dt_search.fit(X_tr_res, y_tr_res)
    dt_best = dt_search.best_estimator_
    joblib.dump(dt_best, MODEL_DIR / 'dt_stage1.joblib')

    mlp = MLPClassifier(random_state=42, early_stopping=True, max_iter=300)
    mlp_params = {'hidden_layer_sizes':[(50,),(100,)], 'alpha':[1e-4,1e-3]}
    mlp_search = GridSearchCV(mlp, mlp_params, scoring='average_precision', cv=3, n_jobs=-1)
    mlp_search.fit(X_tr_res, y_tr_res)
    mlp_best = mlp_search.best_estimator_
    joblib.dump(mlp_best, MODEL_DIR / 'mlp_stage1.joblib')

    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_tr_res, y_tr_res)
    joblib.dump(svc, MODEL_DIR / 'svc_stage1.joblib')

    def evaluate(model, X_t, y_true):
        probs = model.predict_proba(X_t)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_t)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan')
        auprc = average_precision_score(y_true, probs)
        pred = model.predict(X_t)
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, pred)
        return dict(auc=auc, auprc=auprc, accuracy=acc, precision=p, recall=r, f1=f)

    res = {}
    res['dt_val'] = evaluate(dt_best, X_val_t, y_val)
    res['mlp_val'] = evaluate(mlp_best, X_val_t, y_val)


    df_test = df.loc[test_ids].copy()
    y_test = labels.reindex(df_test.index).fillna(0).astype(int)
    X_test_t = pre.transform(df_test)
    res['dt_test'] = evaluate(dt_best, X_test_t, y_test)
    res['mlp_test'] = evaluate(mlp_best, X_test_t, y_test)
    res['svc_test'] = evaluate(svc, X_test_t, y_test)

    joblib.dump(res, MODEL_DIR / 'results_stage1.joblib')
    rows = []
    for key, metrics in res.items():
        model_name = key.replace('_val', '').replace('_test', '').upper()
        split_name = 'Dataset 1 validation' if key.endswith('_val') else 'Dataset 2 test'
        rows.append({
            'stage': 'stage1',
            'model': model_name,
            'evaluation_split': split_name,
            'roc_auc': metrics.get('auc'),
            'auprc': metrics.get('auprc'),
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1': metrics.get('f1'),
        })
    pd.DataFrame(rows).to_csv(REPORT_DIR / 'models_metrics_stage1.csv', index=False)
    print('Saved stage 1 models and metrics to', MODEL_DIR)
    print(res)


if __name__ == "__main__":
    train_and_eval()
