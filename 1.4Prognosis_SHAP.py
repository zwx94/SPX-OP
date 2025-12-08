# -*- coding: utf-8 -*-
"""
SHAP interpretability for 5-fold binary XGBoost CV (using saved models & splits)

Outputs (under results_dir/shap/):
  per-fold/
    foldXX_shap_values_test.npy
    foldXX_mean_abs_shap.csv
    foldXX_beeswarm_top20.pdf
    foldXX_bar_top20.pdf
    foldXX_dependence_feat{1,2,3}.pdf
  overall/
    oof_mean_abs_shap.csv
    oof_beeswarm_top20.pdf
    oof_bar_top20.pdf

Notes:
- Uses TreeExplainer(model_output="probability", feature_perturbation="interventional")
- Background: up to 1000 rows sampled from that fold's TRAIN set (after imputation)
- Features names: auto f0..f{p-1} unless you provide a list/npz path
"""
# %%
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from joblib import load as joblib_load
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# ========= config =========
RESULTS_DIR    = "xgb_binary_nestedcv_results_AP"   
MODEL_PATTERN  = "xgb_fold{fold:02d}.joblib"     
TOP_N_PLOT     = 20                               
BG_MAX_ROWS    = 1000                             
OOF_MAX_ROWS   = 5000                             
SEED           = 2025
FIGSIZE        = (7, 6)                           
DPI            = 300
BASE_FONTSIZE  = 12
FEATURE_NAMES  = None   
# ===================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE-1,
    "ytick.labelsize": BASE_FONTSIZE-1,
    "legend.fontsize": BASE_FONTSIZE-2,
})

# SHAP
try:
    import shap
except Exception as e:
    raise ImportError("请先安装 SHAP：pip install shap 或 conda install -c conda-forge shap") from e

# --- 工具函数 ---
def discover_folds(folds_dir: Path):
    pat = re.compile(r"fold(\d{2})_train_set\.npz$")
    folds = []
    for p in sorted(folds_dir.glob("fold*_train_set.npz")):
        m = pat.search(p.name)
        if m:
            folds.append(int(m.group(1)))
    if not folds:
        raise FileNotFoundError(f"未在 {folds_dir} 找到 foldXX_train_set.npz")
    return sorted(folds)

def load_model(models_dir: Path, fold: int):
    p = models_dir / MODEL_PATTERN.format(fold=fold)
    if p.exists():
        return joblib_load(p), p
    # 兼容其它命名
    cands = list(models_dir.glob(f"*fold{fold:02d}*.joblib"))
    if cands:
        return joblib_load(cands[0]), cands[0]
    raise FileNotFoundError(f"未找到第 {fold} 折模型（期待 {p} 或 *fold{fold:02d}*.joblib）")

def extract_steps(model):
    """返回 (pipeline, imputer_step, clf_step)。若不是Pipeline则做兜底。"""
    from sklearn.pipeline import Pipeline
    pipe = model
    imputer = None
    clf = None
    if isinstance(model, Pipeline):
        # 常见命名：imputer / clf
        imputer = pipe.named_steps.get("imputer", None)
        clf = pipe.named_steps.get("clf", None)
    else:
        clf = model
    return pipe, imputer, clf

def to_feature_names(p):
    if FEATURE_NAMES is not None:
        return np.asarray(FEATURE_NAMES, dtype=object)
    return np.array([f"f{i}" for i in range(p)], dtype=object)

def shap_summary_plot(shap_values, X, feature_names, title, out_png):
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False, max_display=TOP_N_PLOT, plot_type="dot"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def shap_bar_plot(mean_abs_shap, feature_names, title, out_png):
    # 仅取 top N
    order = np.argsort(-mean_abs_shap)[:TOP_N_PLOT]
    vals  = mean_abs_shap[order]
    names = feature_names[order]
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("mean(|SHAP|)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def shap_dependence_plots(shap_values, X, feature_names, out_dir, top_n=3):
    order = np.argsort(-np.abs(shap_values).mean(axis=0))[:top_n]
    for rank, j in enumerate(order, start=1):
        plt.figure(figsize=FIGSIZE, dpi=DPI)
        shap.dependence_plot(
            j, shap_values, X,
            feature_names=feature_names,
            interaction_index=None, show=False
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"dependence_feat{rank}_({feature_names[j]}).pdf")
        plt.close()

def main(results_dir: str):
    np.random.seed(SEED)

    results_dir = Path(results_dir)
    folds_dir   = results_dir / "folds"
    models_dir  = results_dir / "models"
    shap_dir    = results_dir / "shap"
    (shap_dir / "per_fold").mkdir(parents=True, exist_ok=True)
    (shap_dir / "overall").mkdir(exist_ok=True)

    folds = discover_folds(folds_dir)
    print("发现折：", folds)

    oof_X_list, oof_shap_list = [], []

    for fold in tqdm(folds, desc="Per-fold SHAP"):
        tr = np.load(folds_dir / f"fold{fold:02d}_train_set.npz")
        te = np.load(folds_dir / f"fold{fold:02d}_test_set.npz")
        X_tr, y_tr = tr["X"], tr["y"].astype(int)
        X_te, y_te = te["X"], te["y"].astype(int)
        n, p = X_tr.shape
        feat_names = to_feature_names(p)

        model, model_path = load_model(models_dir, fold)
        pipe, imputer, clf = extract_steps(model)
        if clf is None:
            raise RuntimeError(f"第 {fold} 折未找到分类器步骤 'clf'，请检查模型：{model_path}")

        if imputer is not None:
            X_tr_imp = imputer.transform(X_tr)
            X_te_imp = imputer.transform(X_te)
        else:
            X_tr_imp, X_te_imp = X_tr, X_te

        background = X_tr_imp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(
                clf, data=background,
                feature_perturbation="interventional",
                model_output="probability"
            )

        shap_values = explainer.shap_values(X_te_imp, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        np.save(shap_dir / "per_fold" / f"fold{fold:02d}_shap_values_test.npy", shap_values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        df_imp.to_csv(shap_dir / "per_fold" / f"fold{fold:02d}_mean_abs_shap.csv", index=False, encoding="utf-8")

        shap_summary_plot(
            shap_values, X_te_imp, feat_names,
            title=f"Fold {fold} SHAP Beeswarm (test, top {TOP_N_PLOT})",
            out_png=shap_dir / "per_fold" / f"fold{fold:02d}_beeswarm_top{TOP_N_PLOT}.pdf"
        )
        shap_bar_plot(
            mean_abs, feat_names,
            title=f"Fold {fold} mean(|SHAP|) (test, top {TOP_N_PLOT})",
            out_png=shap_dir / "per_fold" / f"fold{fold:02d}_bar_top{TOP_N_PLOT}.pdf"
        )
        shap_dependence_plots(
            shap_values, X_te_imp, feat_names,
            out_dir=(shap_dir / "per_fold"),
            top_n=3
        )

        oof_X_list.append(X_te_imp)
        oof_shap_list.append(shap_values)

    # ===== Overall (OOF) =====
    X_all   = np.vstack(oof_X_list)
    SHAP_all= np.vstack(oof_shap_list)

    mean_abs_all = np.abs(SHAP_all).mean(axis=0)
    feat_names_all = to_feature_names(X_all.shape[1])

    pd.DataFrame({
        "feature": feat_names_all,
        "mean_abs_shap": mean_abs_all
    }).sort_values("mean_abs_shap", ascending=False)\
     .to_csv(shap_dir / "overall" / "oof_mean_abs_shap.csv", index=False, encoding="utf-8")

    X_plot, SHAP_plot = X_all, SHAP_all

    shap_summary_plot(
        SHAP_plot, X_plot, feat_names_all,
        title=f"OOF SHAP Beeswarm (top {TOP_N_PLOT})",
        out_png=shap_dir / "overall" / f"oof_beeswarm_top{TOP_N_PLOT}.pdf"
    )
    shap_bar_plot(
        mean_abs_all, feat_names_all,
        title=f"OOF mean(|SHAP|) (top {TOP_N_PLOT})",
        out_png=shap_dir / "overall" / f"oof_bar_top{TOP_N_PLOT}.pdf"
    )

    print("\n=== 完成 ===")
    print("SHAP 输出目录：", (shap_dir).resolve())
    print(" - 每折结果：", (shap_dir / "per_fold").resolve())
    print(" - 全集结果：", (shap_dir / "overall").resolve())

# %%
if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    
    # %%
    significant_C0_vs_C1 = pd.read_csv('limma_DE/pairwise/full_C0_vs_C1.csv')
    
    Protein_C0_vs_C1 = significant_C0_vs_C1.iloc[:, 0].values[significant_C0_vs_C1[['adj.P.Val']].values.reshape(-1) < 0.05]

    Protein_DE = np.unique(Protein_C0_vs_C1)

    # %%
    FEATURE_NAMES = Protein_DE.tolist()
    main(RESULTS_DIR)

    # %%