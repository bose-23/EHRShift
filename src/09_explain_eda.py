import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

OUT = Path('outputs')
FIG = OUT/'figures'
FIG.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT/'models'


def load_data():
    df = pd.read_csv(OUT / 'aggregated_patients.csv', dtype={'PATIENT':str}, parse_dates=['last_encounter']).set_index('PATIENT')
    labels = pd.read_csv(OUT / 'diabetes_labels.csv', index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:,0]
    train_ids = pd.read_csv(OUT / 'train_ids.csv')['PATIENT'].tolist()
    test_ids = pd.read_csv(OUT / 'test_ids.csv')['PATIENT'].tolist()
    return df, labels, train_ids, test_ids


def get_feature_names(pre):
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        pass
    try:
        num_cols = pre.transformers_[0][2]
        cat_info = pre.transformers_[1]
        cat_cols = cat_info[2]
        ohe = pre.named_transformers_['cat'].named_steps['ohe']
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        names = list(num_cols) + cat_names
        return names
    except Exception:
        try:
            return list(pre.feature_names_in_)
        except Exception:
            return None


def tree_feature_importance(model, feat_names, topk=20):
    imp = getattr(model, 'feature_importances_', None)
    if imp is None:
        return None
    if feat_names is None:
        feat_names = [f'f{i}' for i in range(len(imp))]
    idx = np.argsort(imp)[::-1][:topk]
    return [(feat_names[i], float(imp[i])) for i in idx]


def pretty_feature_name(name):
    if name.startswith("num__"):
        name = name.replace("num__", "", 1)
    elif name.startswith("cat__"):
        name = name.replace("cat__", "", 1)

    if name.startswith("GENDER_"):
        return f"Gender: {name.split('_', 1)[1]}"
    if name.startswith("RACE_"):
        return f"Race: {name.split('_', 1)[1].title()}"

    return name.replace("_", " ")


def clean_ranked_features(rows, min_value=0.0, topk=15):
    cleaned = []
    for feature, value in rows:
        if value <= min_value:
            continue
        cleaned.append((pretty_feature_name(feature), float(value)))
    return cleaned[:topk]


def run_explain():
    df, labels, train_ids, test_ids = load_data()
    pre = joblib.load(MODEL_DIR / 'preprocessor.joblib')
    feat_names = get_feature_names(pre)
    dt = joblib.load(MODEL_DIR / 'dt_retrained.joblib') if (MODEL_DIR/'dt_retrained.joblib').exists() else None
    mlp = joblib.load(MODEL_DIR / 'mlp_finetuned.joblib') if (MODEL_DIR/'mlp_finetuned.joblib').exists() else None

    X_test = df.loc[test_ids]
    y_test = labels.reindex(test_ids).fillna(0).astype(int)
    X_test_t = pre.transform(X_test)
    dt_imp = []
    perm = []

    if dt is not None:
        dt_imp = tree_feature_importance(dt, feat_names, topk=40)
        dt_imp = clean_ranked_features(dt_imp, min_value=1e-6, topk=15)
        if dt_imp:
            pd.DataFrame(dt_imp, columns=['feature','importance']).to_csv(OUT/'figures'/'dt_feature_importances.csv', index=False)
            df_imp = pd.DataFrame(dt_imp, columns=['feature','importance']).set_index('feature')
            plt.figure(figsize=(8,6))
            sns.barplot(x='importance', y=df_imp.index, data=df_imp.reset_index())
            plt.title('Decision Tree Feature Importances (retrained)')
            plt.tight_layout()
            plt.savefig(FIG/'fi_dt_retrained.png')
            plt.close()

    if mlp is not None:
        print('Computing MLP permutation importance...')
        try:
            X_test_dense = X_test_t.toarray() if hasattr(X_test_t, 'toarray') else X_test_t
        except Exception:
            X_test_dense = X_test_t
        r = permutation_importance(mlp, X_test_dense, y_test, n_repeats=20, random_state=42, scoring='average_precision', n_jobs=-1)
        importances = r.importances_mean
        if feat_names is None:
            feat_names = [f'f{i}' for i in range(len(importances))]
        idx = np.argsort(importances)[::-1][:40]
        perm = [(feat_names[i], float(importances[i])) for i in idx]
        perm = clean_ranked_features(perm, min_value=1e-4, topk=15)
        pd.DataFrame(perm, columns=['feature','perm_importance']).to_csv(OUT/'figures'/'mlp_permutation_importances.csv', index=False)
        df_perm = pd.DataFrame(perm, columns=['feature','perm_importance']).set_index('feature')
        plt.figure(figsize=(8,6))
        sns.barplot(x='perm_importance', y=df_perm.index, data=df_perm.reset_index())
        plt.title('MLP Permutation Importances (Dataset2 test)')
        plt.tight_layout()
        plt.savefig(FIG/'fi_mlp_permutation.png')
        plt.close()

    df['label'] = labels.reindex(df.index).fillna(0).astype(int)
    df['last_encounter'] = pd.to_datetime(df['last_encounter'], errors='coerce')
    df['year_month'] = df['last_encounter'].dt.to_period('M')
    prevalence = df.groupby('year_month')['label'].mean().dropna()
    counts = df.groupby('year_month')['label'].count()
    plt.figure(figsize=(10,4))
    prevalence.plot(marker='o')
    plt.ylabel('Diabetes prevalence')
    plt.title('Label prevalence over time (per month)')
    plt.tight_layout()
    plt.savefig(FIG/'label_prevalence_over_time.png')
    plt.close()

    candidates = ['Body Weight_last','BMI_last','Glucose_last','med_count','cond_count']
    present = [c for c in candidates if c in df.columns]
    if present:
        train_df = df.loc[train_ids]
        test_df = df.loc[test_ids]
        for col in present:
            plt.figure(figsize=(8,4))
            sns.kdeplot(train_df[col].dropna(), label='Dataset1', bw_adjust=1.0)
            sns.kdeplot(test_df[col].dropna(), label='Dataset2', bw_adjust=1.0)
            plt.legend()
            plt.title(f'Distribution comparison: {col}')
            plt.tight_layout()
            plt.savefig(FIG/f'dist_{col}.png')
            plt.close()

    top_feats = None
    if dt_imp:
        top_feats = [f for f,_ in dt_imp[:20]]
    elif perm:
        top_feats = [f for f,_ in perm[:20]]
    if top_feats is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if c in top_feats]
        if cols:
            corr = df[cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Correlation (top numeric features)')
            plt.tight_layout()
            plt.savefig(FIG/'corr_top_numeric.png')
            plt.close()

    summary = []
    if dt_imp:
        summary.append('Top Decision Tree features:')
        summary += [f'{i+1}. {f} ({imp:.4f})' for i,(f,imp) in enumerate(dt_imp[:20])]
    if perm:
        summary.append('\nTop MLP permutation features:')
        summary += [f'{i+1}. {f} ({imp:.4f})' for i,(f,imp) in enumerate(perm[:20])]
    summary_text = '\n'.join(summary)
    Path(OUT/'reports').mkdir(exist_ok=True)
    with open(OUT/'reports'/'feature_importance_summary.txt','w') as fh:
        fh.write(summary_text)
    print('Saved feature importance summary to outputs/reports/feature_importance_summary.txt')


if __name__ == '__main__':
    run_explain()
