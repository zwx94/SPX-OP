# -*- coding: utf-8 -*-
# %%
import os, sys, json, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss, top_k_accuracy_score, confusion_matrix,
    make_scorer
)

from joblib import dump as joblib_dump

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# XGBoost
try:
    import xgboost as xgbpkg
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("install xgboost：pip install xgboost 或 conda install -c conda-forge xgboost") from e

# %%
def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_outdir(out_dir=None) -> Path:
    if out_dir is None:
        out_dir = Path(f"xgb_multiclass_nestedcv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(out_dir)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    (out_dir / "gridcv").mkdir(exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)
    (out_dir / "per_class").mkdir(exist_ok=True)
    return out_dir


def save_fold_data(out_dir: Path, fold: int, Xtr, ytr, Xte, yte, tr_idx, te_idx):
    np.save(out_dir / "folds" / f"fold{fold:02d}_train_idx.npy", tr_idx)
    np.save(out_dir / "folds" / f"fold{fold:02d}_test_idx.npy", te_idx)
    np.savez_compressed(out_dir / "folds" / f"fold{fold:02d}_train_set.npz",
                        X=Xtr.astype(np.float32), y=ytr.astype(np.int8))
    np.savez_compressed(out_dir / "folds" / f"fold{fold:02d}_test_set.npz",
                        X=Xte.astype(np.float32), y=yte.astype(np.int8))


def multiclass_pr_aucs(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray):
    y_bin = label_binarize(y_true, classes=classes)
    pr_auc_per_class = []
    for i in range(len(classes)):
        p, r, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        pr_auc_per_class.append(auc(r, p))
    pr_auc_per_class = np.array(pr_auc_per_class)
    pr_auc_macro = float(np.nanmean(pr_auc_per_class))
    # micro
    p_mi, r_mi, _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
    pr_auc_micro = float(auc(r_mi, p_mi))
    # Average Precision（AP）
    ap_macro = average_precision_score(y_bin, proba, average="macro")
    ap_micro = average_precision_score(y_bin, proba, average="micro")
    ap_weighted = average_precision_score(y_bin, proba, average="weighted")
    return dict(
        pr_auc_macro=pr_auc_macro, pr_auc_micro=pr_auc_micro, pr_auc_per_class=pr_auc_per_class,
        ap_macro=ap_macro, ap_micro=ap_micro, ap_weighted=ap_weighted
    )


def evaluate_metrics(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray, classes: np.ndarray):
    mets = {}

    try:
        mets["roc_auc_ovr_macro"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        mets["roc_auc_ovr_weighted"] = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except Exception:
        mets["roc_auc_ovr_macro"] = np.nan
        mets["roc_auc_ovr_weighted"] = np.nan
    try:
        mets["roc_auc_ovo_macro"] = roc_auc_score(y_true, proba, multi_class="ovo", average="macro")
    except Exception:
        mets["roc_auc_ovo_macro"] = np.nan

    pr = multiclass_pr_aucs(y_true, proba, classes)
    mets.update({
        "pr_auc_macro": pr["pr_auc_macro"],
        "pr_auc_micro": pr["pr_auc_micro"],
        "avg_precision_macro": pr["ap_macro"],
        "avg_precision_micro": pr["ap_micro"],
        "avg_precision_weighted": pr["ap_weighted"],
    })

    mets["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mets["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    mets["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    mets["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mets["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    mets["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    mets["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    mets["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    mets["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    mets["accuracy"] = accuracy_score(y_true, y_pred)
    mets["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    mets["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    mets["mcc"] = matthews_corrcoef(y_true, y_pred)

    try:
        mets["log_loss"] = log_loss(y_true, proba, labels=classes)
    except Exception:
        mets["log_loss"] = np.nan

    k2, k3 = min(2, len(classes)), min(3, len(classes))
    mets["top2_accuracy"] = top_k_accuracy_score(y_true, proba, k=k2, labels=classes)
    mets["top3_accuracy"] = top_k_accuracy_score(y_true, proba, k=k3, labels=classes)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    mets["_cm"] = cm
    mets["_pr_auc_per_class"] = pr["pr_auc_per_class"]
    return mets


def build_xgb(seed=42, device="cpu"):
    return XGBClassifier(
        booster="gbtree",
        max_depth=10,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        subsample=0.8,
        sampling_method="uniform",
        min_child_weight=50,
        colsample_bytree=0.8,
        # n_jobs=-1,
        random_state=seed,
        tree_method="hist",
        device=device, 
    )


def run_nested_cv_xgb(
    protein_exp: np.ndarray,
    label: np.ndarray,
    seed: int = 42,
    n_splits_outer: int = 5,
    n_splits_inner: int = 5,
    out_dir: str | Path | None = None,
    device: str = "cpu",   # 若用GPU，设 "cuda"
):
    set_global_seed(seed)
    t0_all = time.perf_counter()

    X = np.asarray(protein_exp).astype(np.float32, copy=False)
    y = np.asarray(label).astype(int)
    assert X.ndim == 2, "protein_exp dim error"
    assert set(np.unique(y)).issubset({0, 1, 2}), "label error"
    assert X.shape[0] == len(y), "dimension error"

    classes = np.array(sorted(np.unique(y)))
    out_dir = ensure_outdir(out_dir)

    try:
        import sklearn
        vers = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgbpkg.__version__,
            "device": device,
        }
    except Exception:
        vers = {"python": sys.version.split()[0], "device": device}
    with open(out_dir / "versions.json", "w", encoding="utf-8") as f:
        json.dump(vers, f, ensure_ascii=False, indent=2)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_counts": {int(c): int((y == c).sum()) for c in classes},
            "outer_cv": n_splits_outer,
            "inner_cv": n_splits_inner,
            "seed": seed
        }, f, ensure_ascii=False, indent=2)

    skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

    metrics_rows, time_rows = [], []
    fold_bar = tqdm(range(1, n_splits_outer + 1), desc="Outer CV folds", leave=True)

    for fold, (tr_idx, te_idx) in zip(fold_bar, skf_outer.split(X, y)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        save_fold_data(out_dir, fold, Xtr, ytr, Xte, yte, tr_idx, te_idx)

        pipe = Pipeline([
            ("clf", build_xgb(seed=seed, device=device))
        ])

        grid_params = {
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "clf__n_estimators":  [50, 100, 200, 300, 500, 700, 900, 1000],
        }

        skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)

        t0_grid = time.perf_counter()
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=grid_params,
            scoring='average_precision',     
            cv=skf_inner,
            n_jobs=32,
            refit=False,
            verbose=1               
        )
        grid.fit(Xtr, ytr)
        grid_time = time.perf_counter() - t0_grid

        best_params = grid.best_params_
        pd.DataFrame(grid.cv_results_).to_csv(out_dir / "gridcv" / f"fold{fold:02d}_cv_results.csv",
                                              index=False, encoding="utf-8")
        with open(out_dir / "gridcv" / f"fold{fold:02d}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        pipe_best = Pipeline([
            ("clf", build_xgb(seed=seed, device=device))
        ])
        pipe_best.set_params(**best_params)

        t0_fit = time.perf_counter()
        pipe_best.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0_fit

        t0_pred = time.perf_counter()
        proba = pipe_best.predict_proba(Xte)      # (n, 3)
        ypred = np.asarray(classes)[np.argmax(proba, axis=1)]
        pred_time = time.perf_counter() - t0_pred

        mets = evaluate_metrics(yte, proba, ypred, classes=classes)
        row = {"fold": fold, **{k: v for k, v in mets.items() if not k.startswith("_")}}
        metrics_rows.append(row)

        pred_df = pd.DataFrame({"index": te_idx, "y_true": yte, "y_pred": ypred})
        for i, c in enumerate(classes):
            pred_df[f"prob_class_{c}"] = proba[:, i]
        pred_df.to_csv(out_dir / "preds" / f"xgb_fold{fold:02d}_pred.csv", index=False, encoding="utf-8")

        cm = mets["_cm"]
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
        cm_df.to_csv(out_dir / "per_class" / f"xgb_fold{fold:02d}_confusion_matrix.csv", encoding="utf-8")

        prc = mets["_pr_auc_per_class"]
        pc_df = pd.DataFrame({"class": classes, "pr_auc_trapz": prc})
        pc_df.to_csv(out_dir / "per_class" / f"xgb_fold{fold:02d}_pr_auc_per_class.csv", index=False, encoding="utf-8")

        joblib_dump(pipe_best, out_dir / "models" / f"xgb_fold{fold:02d}.joblib")

        time_rows.append({
            "fold": fold,
            "grid_search_time_sec": grid_time,
            "fit_time_sec": fit_time,
            "predict_time_sec": pred_time,
            "total_fold_time_sec": grid_time + fit_time + pred_time
        })

    met_df = pd.DataFrame(metrics_rows)
    met_df.to_csv(out_dir / "metrics_per_fold.csv", index=False, encoding="utf-8")

    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(out_dir / "time_per_fold.csv", index=False, encoding="utf-8")

    def mean_std(s): return f"{np.nanmean(s):.4f} ± {np.nanstd(s):.4f}"
    summary = met_df.drop(columns=["fold"]).agg(mean_std).to_frame("mean±std")
    summary.to_csv(out_dir / "metrics_summary.csv", encoding="utf-8")

    total_sec = time.perf_counter() - t0_all
    with open(out_dir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"Total runtime (seconds): {total_sec:.2f}\n")

    print("\n=== done ===")

# %%
if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    
    SEED = 2025
    N_SPLITS = 5

    # %%
    biomarker_info_prognosis = pd.read_csv(current_path + '/../v13_prognosis/biomarker_analysis/importance_overall.csv')
    biomarker_names_prognosis = biomarker_info_prognosis['x'].tolist()

    biomarker_info_diagnosis = pd.read_csv(current_path + '/../v12_diagnosis/biomarker_analysis/importance_overall.csv')
    biomarker_names_diagnosis = biomarker_info_diagnosis['x'].tolist()

    biomarker_names = []
    biomarker_names.extend(biomarker_names_prognosis)
    biomarker_names.extend(biomarker_names_diagnosis)

    biomarker_names = list(set(biomarker_names))

    # %%
    df_balanced = pd.read_csv(current_path + '/data/df_balanced.csv')

    protein_exp_all = df_balanced.iloc[:, 9:][biomarker_names].values

    label_all = df_balanced['label_0_1_2'].values

    # %%
    labels = label_all.copy()
    protein_exp = protein_exp_all.copy()

    # %%
    run_nested_cv_xgb(
        protein_exp=protein_exp,
        label=labels,
        seed=SEED,
        n_splits_outer=5,
        n_splits_inner=5,
        out_dir="xgb_multiclass_nestedcv_results_ap_prognosis_diagnosis",
        device="cuda",          
    )

    # %%