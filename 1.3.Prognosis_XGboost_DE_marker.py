# -*- coding: utf-8 -*-
"""
Binary classification with XGBoost on proteomics:
- Outer 5-fold CV for unbiased evaluation
- Inner 5-fold GridSearchCV per outer fold (eta & nrounds)
- Fixed XGB params per your spec (gbtree, depth=10, etc.)
- Saves: per-fold train/test sets, models, predictions, metrics, CV results
- Metrics: ROC-AUC, PR-AUC (AUPRC), Average Precision, Precision/Recall/F1,
           Accuracy, Balanced Acc, Specificity, Youden J, Kappa, MCC, LogLoss, Brier
- Reproducible seeds, tqdm progress, runtime stats
"""
# %%
import os, sys, json, time, random, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, brier_score_loss, confusion_matrix
)
from joblib import dump as joblib_dump

# 进度条
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# XGBoost
try:
    import xgboost as xgbpkg
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("需要安装 xgboost：pip install xgboost 或 conda install -c conda-forge xgboost") from e

# %%
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_outdir(out_dir=None) -> Path:
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"xgb_binary_nestedcv_{stamp}")
    out_dir = Path(out_dir)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)
    (out_dir / "gridcv").mkdir(exist_ok=True)
    return out_dir


def save_fold_data(out_dir: Path, fold: int, X_tr, y_tr, X_te, y_te, tr_idx, te_idx):
    fp = out_dir / "folds"
    np.save(fp / f"fold{fold:02d}_train_idx.npy", tr_idx)
    np.save(fp / f"fold{fold:02d}_test_idx.npy", te_idx)
    np.savez_compressed(fp / f"fold{fold:02d}_train_set.npz",
                        X=X_tr.astype(np.float32), y=y_tr.astype(np.int8))
    np.savez_compressed(fp / f"fold{fold:02d}_test_set.npz",
                        X=X_te.astype(np.float32), y=y_te.astype(np.int8))


def pr_auc_from_scores(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return auc(r, p)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0   # TPR, Recall+
    spec = tn / (tn + fp) if (tn + fp) else 0.0   # TNR, Specificity
    youden = sens + spec - 1

    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan

    metrics["pr_auc"] = pr_auc_from_scores(y_true, y_prob)                   # AUPRC
    metrics["avg_precision"] = average_precision_score(y_true, y_prob)       # AP

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["specificity"] = spec
    metrics["youden_j"] = youden
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    try:
        metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        metrics["log_loss"] = np.nan
    try:
        metrics["brier"] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics["brier"] = np.nan

    return metrics, y_pred


def build_xgb(seed: int, device: str = "cpu", spw: float = 1.0):
    params = dict(
        max_depth=6,
        eval_metric="logloss",
        objective="binary:logistic",
        subsample=0.8,
        reg_lambda=1.0,
        colsample_bytree=0.8,      
        tree_method="hist",
        random_state=seed,
        # n_jobs=-1,
        device=device,             # 'cpu' 或 'cuda'
        scale_pos_weight=spw,
    )
    return XGBClassifier(**params)


def run_xgb_nested_cv(
    protein_exp: np.ndarray,
    label: np.ndarray,
    seed: int = 42,
    n_splits_outer: int = 5,
    n_splits_inner: int = 5,
    out_dir: str | Path | None = None,
    device: str = "cpu",   
):
    set_global_seed(seed)
    t0_total = time.perf_counter()

    X = np.asarray(protein_exp)
    y = np.asarray(label).astype(int)

    spw = float(int(np.sum(y ==0)/max(int(np.sum(y == 1)), 1))) if int(np.sum(y == 1)) > 0 else 1.0

    assert X.ndim == 2, "protein_exp dim error"
    assert set(np.unique(y)).issubset({0, 1}), "label must be 0/1"
    assert X.shape[0] == y.shape[0], "dimmension error"

    # 转 float32（更省内存）
    X = X.astype(np.float32, copy=False)

    out_dir = ensure_outdir(out_dir)

    # 记录环境信息
    with open(out_dir / "versions.json", "w", encoding="utf-8") as f:
        try:
            import sklearn
            vers = {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgbpkg.__version__,
            }
        except Exception:
            vers = {"python": sys.version.split()[0]}
        json.dump(vers, f, ensure_ascii=False, indent=2)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        meta = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "pos_count": int((y == 1).sum()),
            "neg_count": int((y == 0).sum()),
            "seed": seed,
            "outer_cv": n_splits_outer,
            "inner_cv": n_splits_inner,
            "device": device,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        json.dump(meta, f, ensure_ascii=False, indent=2)

    skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

    metrics_rows = []
    time_rows = []

    fold_bar = tqdm(range(1, n_splits_outer + 1), desc="Outer CV folds", leave=True)

    for fold, (tr_idx, te_idx) in zip(fold_bar, skf_outer.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        save_fold_data(out_dir, fold, X_tr, y_tr, X_te, y_te, tr_idx, te_idx)

        pipe = Pipeline([
            ("clf", build_xgb(seed=seed, device=device, spw=spw)),
        ])

        param_grid = {
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "clf__n_estimators":  [50, 100, 200, 300, 500, 700, 900, 1000],
        }

        skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)

        t0_grid = time.perf_counter()
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="average_precision",
            cv=skf_inner,
            n_jobs=64,
            # n_jobs=-1,
            refit=False,
            verbose=1  
        )
        grid.fit(X_tr, y_tr)
        t_grid = time.perf_counter() - t0_grid

        best_params = grid.best_params_
        cvdf = pd.DataFrame(grid.cv_results_)
        cvdf.to_csv(out_dir / "gridcv" / f"fold{fold:02d}_cv_results.csv", index=False, encoding="utf-8")
        with open(out_dir / "gridcv" / f"fold{fold:02d}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        pipe_best = Pipeline([
            ("clf", build_xgb(seed=seed, device=device, spw=spw)),
        ])
        pipe_best.set_params(**best_params)

        t0_fit = time.perf_counter()
        pipe_best.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0_fit

        y_prob = pipe_best.predict_proba(X_te)[:, 1]
        met, y_pred = binary_metrics(y_te, y_prob, threshold=0.5)

        pred_df = pd.DataFrame({
            "index": te_idx,
            "y_true": y_te,
            "y_prob": y_prob,
            "y_pred": y_pred
        })
        pred_df.to_csv(out_dir / "preds" / f"xgb_fold{fold:02d}_pred.csv", index=False, encoding="utf-8")

        model_path = out_dir / "models" / f"xgb_fold{fold:02d}.joblib"
        joblib_dump(pipe_best, model_path)

        row = {"fold": fold, **met}
        metrics_rows.append(row)

        time_rows.append({
            "fold": fold,
            "grid_search_time_sec": t_grid,
            "refit_time_sec": t_fit,
            "total_fold_time_sec": t_grid + t_fit
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_per_fold.csv", index=False, encoding="utf-8")

    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(out_dir / "time_per_fold.csv", index=False, encoding="utf-8")

    def mean_std(s): return f"{np.nanmean(s):.4f} ± {np.nanstd(s):.4f}"
    summary = metrics_df.drop(columns=["fold"]).agg(mean_std).to_frame("mean±std")
    summary.to_csv(out_dir / "metrics_summary.csv", encoding="utf-8")

    elapsed = time.perf_counter() - t0_total
    with open(out_dir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"Total runtime (seconds): {elapsed:.2f}\n")

    print("\n=== done ===")


# %%
if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    
    SEED = 2025
    N_SPLITS = 5

    # %%
    # read DE results
    significant_C0_vs_C1 = pd.read_csv(current_path + '/limma_DE/pairwise/full_C0_vs_C1.csv')
    
    Protein_C0_vs_C1 = significant_C0_vs_C1.iloc[:, 0].values[significant_C0_vs_C1[['adj.P.Val']].values.reshape(-1) < 0.05]

    Protein_DE = np.unique(Protein_C0_vs_C1)

    # %%
    # import data
    df_balanced = pd.read_csv(current_path + '/data/df_balanced.csv')

    protein_exp_all = df_balanced.iloc[:, 9:][Protein_DE].values

    label_all = df_balanced['label_0_1'].values

    # %%
    labels = label_all.copy()
    protein_exp = protein_exp_all.copy()

    # %%
    run_xgb_nested_cv(
        protein_exp=protein_exp,
        label=labels,
        seed=SEED,
        n_splits_outer=5,
        n_splits_inner=5,
        out_dir="xgb_binary_nestedcv_results_AP",
        device="cuda"     
    )

    # %%
