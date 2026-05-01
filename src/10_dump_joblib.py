import joblib
import json
from pathlib import Path
import numpy as np
from sklearn.base import is_classifier

MODELS_DIR = Path('outputs/models')
OUT = Path('outputs/reports')
OUT.mkdir(exist_ok=True)


def to_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        return None


def summarize_model(obj):
    out = {}
    try:
        params = obj.get_params()
        out['params'] = to_serializable(params)
    except Exception:
        out['params'] = None
    try:
        fi = getattr(obj, 'feature_importances_', None)
        if fi is not None:
            out['feature_importances'] = to_serializable(fi)
    except Exception:
        pass
    try:
        coef = getattr(obj, 'coef_', None)
        if coef is not None:
            out['coef'] = to_serializable(coef)
    except Exception:
        pass
    return out


def main():
    summaries = {}
    for f in MODELS_DIR.glob('*.joblib'):
        name = f.name
        try:
            obj = joblib.load(f)
        except Exception as e:
            summaries[name] = {'error': str(e)}
            continue
        if isinstance(obj, dict):
            summaries[name] = to_serializable(obj)
        else:
            summaries[name] = summarize_model(obj)
    outf = OUT / 'models_summary.json'
    with open(outf, 'w') as fh:
        json.dump(summaries, fh)
    print('Saved', outf)


if __name__ == '__main__':
    main()
