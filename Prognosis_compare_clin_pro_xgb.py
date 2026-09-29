# -*- coding: utf-8 -*-
"""
Compare three binary XGBoost models for osteoporosis classification:
1) clinical_only
2) proteomics_only
3) clinical_plus_proteomics

Output:
- nested CV metrics for each model
- OOF predictions for each model
- one combined ROC-AUC figure
"""

# %%
import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, brier_score_loss, confusion_matrix
)

from joblib import dump as joblib_dump

# XGBoost
try:
    import xgboost as xgbpkg
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("需要安装 xgboost：pip install xgboost 或 conda install -c conda-forge xgboost") from e

# tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# %%
# =========================================================
# Utils
# =========================================================
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_outdir(out_dir=None) -> Path:
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"xgb_compare_{stamp}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pr_auc_from_scores(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return auc(r, p)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    youden = sens + spec - 1

    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan

    metrics["pr_auc"] = pr_auc_from_scores(y_true, y_prob)
    metrics["avg_precision"] = average_precision_score(y_true, y_prob)
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
        device=device,
        scale_pos_weight=spw,
        n_jobs=1
    )
    return XGBClassifier(**params)


def build_preprocessor(categorical_features, numeric_clinical_features, proteomic_features):
    """
    Dynamic ColumnTransformer:
    - only add transformers that actually have columns
    """
    transformers = []

    if len(categorical_features) > 0:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features)
        )

    if len(numeric_clinical_features) > 0:
        transformers.append(
            ("num", "passthrough", numeric_clinical_features)
        )

    if len(proteomic_features) > 0:
        transformers.append(
            ("prot", "passthrough", proteomic_features)
        )

    if len(transformers) == 0:
        raise ValueError("当前模型没有可用特征。")

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocess


# =========================================================
# Core function: run one model config
# =========================================================
def run_one_model_config(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_features: list,
    numeric_clinical_features: list,
    proteomic_features: list,
    outer_splits,
    seed: int,
    n_splits_inner: int,
    out_dir: Path,
    device: str = "cpu",
    threshold: float = 0.5,
    param_grid: dict | None = None,
):
    """
    Run nested CV for one feature configuration.
    """
    model_dir = out_dir / model_name
    (model_dir / "models").mkdir(parents=True, exist_ok=True)
    (model_dir / "preds").mkdir(exist_ok=True)
    (model_dir / "gridcv").mkdir(exist_ok=True)

    if param_grid is None:
        param_grid = {
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "clf__n_estimators":  [50, 100, 200, 300, 500, 700, 900, 1000],
        }

    spw = float(int(np.sum(y == 0) / max(int(np.sum(y == 1)), 1))) if int(np.sum(y == 1)) > 0 else 1.0

    metrics_rows = []
    oof_rows = []

    for fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        preprocess = build_preprocessor(
            categorical_features=categorical_features,
            numeric_clinical_features=numeric_clinical_features,
            proteomic_features=proteomic_features
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("clf", build_xgb(seed=seed, device=device, spw=spw)),
        ])

        skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="average_precision",
            cv=skf_inner,
            n_jobs=8,
            refit=False,
            verbose=1
        )
        grid.fit(X_tr, y_tr)

        best_params = grid.best_params_

        pd.DataFrame(grid.cv_results_).to_csv(
            model_dir / "gridcv" / f"fold{fold:02d}_cv_results.csv",
            index=False,
            encoding="utf-8"
        )
        with open(model_dir / "gridcv" / f"fold{fold:02d}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        pipe_best = Pipeline([
            ("preprocess", build_preprocessor(
                categorical_features=categorical_features,
                numeric_clinical_features=numeric_clinical_features,
                proteomic_features=proteomic_features
            )),
            ("clf", build_xgb(seed=seed, device=device, spw=spw)),
        ])
        pipe_best.set_params(**best_params)
        pipe_best.fit(X_tr, y_tr)

        y_prob = pipe_best.predict_proba(X_te)[:, 1]
        met, y_pred = binary_metrics(y_te, y_prob, threshold=threshold)

        pred_df = pd.DataFrame({
            "index": te_idx,
            "fold": fold,
            "y_true": y_te,
            "y_prob": y_prob,
            "y_pred": y_pred
        })
        pred_df.to_csv(model_dir / "preds" / f"{model_name}_fold{fold:02d}_pred.csv", index=False, encoding="utf-8")
        oof_rows.append(pred_df)

        joblib_dump(pipe_best, model_dir / "models" / f"{model_name}_fold{fold:02d}.joblib")

        metrics_rows.append({"fold": fold, **met})

    # fold metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(model_dir / "metrics_per_fold.csv", index=False, encoding="utf-8")

    # summary
    metric_cols = [c for c in metrics_df.columns if c != "fold"]
    summary_rows = []
    for c in metric_cols:
        summary_rows.append({
            "metric": c,
            "mean": metrics_df[c].mean(),
            "std": metrics_df[c].std()
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(model_dir / "metrics_summary.csv", index=False, encoding="utf-8")

    # OOF
    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True).sort_values("index")
    oof_df.to_csv(model_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    return {
        "model_name": model_name,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "oof_df": oof_df
    }


# =========================================================
# Plot ROC comparison
# =========================================================
def plot_compare_roc(results: list, out_dir: Path):
    plt.figure(figsize=(7.5, 6.5))

    summary_rows = []

    for res in results:
        model_name = res["model_name"]
        oof_df = res["oof_df"]
        y_true = oof_df["y_true"].values
        y_prob = oof_df["y_prob"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        fold_auc_mean = res["metrics_df"]["roc_auc"].mean()
        fold_auc_std = res["metrics_df"]["roc_auc"].std()

        plt.plot(
            fpr, tpr, lw=2,
            label=f"{model_name} (OOF AUC = {auc_val:.3f}, fold mean = {fold_auc_mean:.3f}±{fold_auc_std:.3f})"
        )

        summary_rows.append({
            "model": model_name,
            "oof_auc": auc_val,
            "fold_auc_mean": fold_auc_mean,
            "fold_auc_std": fold_auc_std
        })

    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC comparison: clinical vs proteomic vs combined")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_dir / "roc_compare_3models.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "roc_compare_3models.pdf", bbox_inches="tight")
    plt.close()

    pd.DataFrame(summary_rows).to_csv(out_dir / "roc_compare_3models_summary.csv", index=False)

# %%
# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    current_path = os.path.dirname(__file__)

    # %%
    SEED = 2025
    N_SPLITS = 5
    DEVICE = "cuda:1"   # 没GPU可改成 "cpu"

    set_global_seed(SEED)
    out_dir = ensure_outdir("prognosis_xgb_compare_3models_proall")

    # 1) read DE proteins
    significant_C0_vs_C1 = pd.read_csv(
        current_path + '/data_prognosis/limma_DE/pairwise/full_C0_vs_C1.csv'
    )
    # Protein_C0_vs_C1 = significant_C0_vs_C1.iloc[:, 0].values[
    #     significant_C0_vs_C1[['adj.P.Val']].values.reshape(-1) < 0.05
    # ]
    # Protein_DE = np.unique(Protein_C0_vs_C1)

    Protein_DE = np.unique(significant_C0_vs_C1.iloc[:, 0])

    # 2) read data
    df_balanced = pd.read_csv(
        current_path + '/data_prognosis/dataset1_OS_0to10_agebin5_nonoverlap.csv',
        index_col=0
    )

    # 3) feature groups
    categorical_features = [
        'Sex',
        'Smoking status | Instance 0',
        'IPAQ activity group | Instance 0',
        'Diabetes diagnosed by doctor | Instance 0',
        'M05_before_assessment',
        'M06_before_assessment'
    ]
    numeric_clinical_features = [
        'Body mass index (BMI) | Instance 0'
    ]
    proteomic_features = Protein_DE.tolist()

    clinical_features = categorical_features + numeric_clinical_features
    all_feature = clinical_features + proteomic_features

    # 4) build full raw X
    X = df_balanced[all_feature].copy()
    y = df_balanced["label_0_1"].values.astype(int)

    print("Data shape:", X.shape)
    print("Positive ratio:", y.mean())
    print("Clinical features:", len(clinical_features))
    print("Proteomic features:", len(proteomic_features))

    # 5) fixed outer splits for fair comparison
    skf_outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    outer_splits = list(skf_outer.split(X, y))

    # %%
    # 6) run three models
    results = []

    # clinical only
    results.append(
        run_one_model_config(
            model_name="clinical_only",
            X=X[clinical_features],
            y=y,
            categorical_features=categorical_features,
            numeric_clinical_features=numeric_clinical_features,
            proteomic_features=[],
            outer_splits=outer_splits,
            seed=SEED,
            n_splits_inner=N_SPLITS,
            out_dir=out_dir,
            device=DEVICE
        )
    )

    # proteomics only
    results.append(
        run_one_model_config(
            model_name="proteomics_only",
            X=X[proteomic_features],
            y=y,
            categorical_features=[],
            numeric_clinical_features=[],
            proteomic_features=proteomic_features,
            outer_splits=outer_splits,
            seed=SEED,
            n_splits_inner=N_SPLITS,
            out_dir=out_dir,
            device=DEVICE
        )
    )

    # clinical + proteomics
    results.append(
        run_one_model_config(
            model_name="clinical_plus_proteomics",
            X=X[all_feature],
            y=y,
            categorical_features=categorical_features,
            numeric_clinical_features=numeric_clinical_features,
            proteomic_features=proteomic_features,
            outer_splits=outer_splits,
            seed=SEED,
            n_splits_inner=N_SPLITS,
            out_dir=out_dir,
            device=DEVICE
        )
    )

    # 7) combined ROC plot
    plot_compare_roc(results, out_dir)

    print("\n=== done ===")
    print(f"Results saved in: {out_dir}")

    # %%