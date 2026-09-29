# -*- coding: utf-8 -*-
"""
Draw fold-wise ROC curves from saved multiclass XGBoost CV results.

Input:
    xgb_three_class_demo_results/all_features/preds/*.csv
    xgb_three_class_demo_results/shap_union_top170_180/preds/*.csv

Output:
    foldwise_macro_roc.png / pdf / tiff
    foldwise_class_roc.png / pdf / tiff
"""
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc


# =========================================================
# 1. 修改这里
# =========================================================
RESULT_DIR = Path("xgb_three_class_demo_results")

MODEL_NAMES = [
    "all_features",
    "shap_union_top170_180"
]

CLASS_LABELS = np.array([0, 1, 2])

CLASS_NAME_MAP = {
    0: "Normal",
    1: "Prevalent OP",
    2: "Incident OP"
}


# =========================================================
# 2. 读取每折预测结果
# =========================================================
def load_fold_predictions(model_dir: Path, model_name: str):
    pred_dir = model_dir / "preds"
    pred_files = sorted(pred_dir.glob(f"{model_name}_fold*_pred.csv"))

    if len(pred_files) == 0:
        raise FileNotFoundError(f"No fold prediction files found in: {pred_dir}")

    fold_dfs = []

    for fp in pred_files:
        df = pd.read_csv(fp)

        required_cols = {"fold", "y_true"}
        prob_cols = {f"prob_class_{c}" for c in CLASS_LABELS}

        missing = required_cols.union(prob_cols) - set(df.columns)
        if len(missing) > 0:
            raise ValueError(f"{fp} missing columns: {missing}")

        fold_dfs.append(df)

    all_df = pd.concat(fold_dfs, axis=0, ignore_index=True)

    return all_df


# =========================================================
# 3. 计算某一折的 macro-average ROC
# =========================================================
def compute_macro_roc(y_true, y_prob, class_labels):
    y_true_bin = label_binarize(y_true, classes=class_labels)

    fpr_dict = {}
    tpr_dict = {}

    for i, c in enumerate(class_labels):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        fpr_dict[c] = fpr_i
        tpr_dict[c] = tpr_i

    all_fpr = np.unique(
        np.concatenate([fpr_dict[c] for c in class_labels])
    )

    mean_tpr = np.zeros_like(all_fpr)

    for c in class_labels:
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])

    mean_tpr /= len(class_labels)
    macro_auc = auc(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, macro_auc


# =========================================================
# 4. 绘制每折 macro-average ROC 曲线
# =========================================================
def plot_foldwise_macro_roc(model_dir: Path, model_name: str, df: pd.DataFrame):
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    prob_cols = [f"prob_class_{c}" for c in CLASS_LABELS]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    fold_auc_rows = []

    # 每折 macro ROC
    for fold_id, subdf in df.groupby("fold"):
        y_true = subdf["y_true"].values.astype(int)
        y_prob = subdf[prob_cols].values.astype(float)

        fpr_macro, tpr_macro, macro_auc = compute_macro_roc(
            y_true=y_true,
            y_prob=y_prob,
            class_labels=CLASS_LABELS
        )

        ax.plot(
            fpr_macro,
            tpr_macro,
            lw=1.8,
            alpha=0.75,
            label=f"Fold {int(fold_id)} (macro AUC = {macro_auc:.3f})"
        )

        fold_auc_rows.append({
            "fold": int(fold_id),
            "macro_auc": macro_auc
        })

    # pooled OOF macro ROC
    y_true_all = df["y_true"].values.astype(int)
    y_prob_all = df[prob_cols].values.astype(float)

    fpr_oof, tpr_oof, oof_macro_curve_auc = compute_macro_roc(
        y_true=y_true_all,
        y_prob=y_prob_all,
        class_labels=CLASS_LABELS
    )

    oof_macro_auc = roc_auc_score(
        y_true_all,
        y_prob_all,
        multi_class="ovr",
        average="macro",
        labels=CLASS_LABELS
    )

    oof_weighted_auc = roc_auc_score(
        y_true_all,
        y_prob_all,
        multi_class="ovr",
        average="weighted",
        labels=CLASS_LABELS
    )

    ax.plot(
        fpr_oof,
        tpr_oof,
        color="black",
        lw=3.0,
        linestyle="--",
        label=f"OOF macro ROC (AUC = {oof_macro_auc:.3f})"
    )

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

    fold_auc_df = pd.DataFrame(fold_auc_rows).sort_values("fold")
    fold_mean = fold_auc_df["macro_auc"].mean()
    fold_std = fold_auc_df["macro_auc"].std()

    text_lines = [
        f"Overall OOF macro AUC: {oof_macro_auc:.3f}",
        f"Overall OOF weighted AUC: {oof_weighted_auc:.3f}",
        f"Fold macro AUC: {fold_mean:.3f} ± {fold_std:.3f}",
    ]

    ax.text(
        0.55,
        0.08,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="gray",
            alpha=0.90
        )
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")

    if model_name == "all_features":
        title = "Fold-wise Macro ROC Curves of All-feature XGBoost"
    elif model_name == "shap_union_top170_180":
        title = "Fold-wise Macro ROC Curves of SHAP-selected XGBoost"
    else:
        title = f"Fold-wise Macro ROC Curves of {model_name}"

    ax.set_title(title, fontsize=13.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_prefix = plot_dir / f"{model_name}_foldwise_macro_roc"

    plt.savefig(str(out_prefix) + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close()

    fold_auc_df.to_csv(
        plot_dir / f"{model_name}_foldwise_macro_auc.csv",
        index=False
    )

    print(f"Saved: {out_prefix}.png")


# =========================================================
# 5. 绘制每一类的每折 ROC 曲线
# =========================================================
def plot_foldwise_class_roc(model_dir: Path, model_name: str, df: pd.DataFrame):
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    prob_cols = [f"prob_class_{c}" for c in CLASS_LABELS]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(
        1,
        len(CLASS_LABELS),
        figsize=(6.2 * len(CLASS_LABELS), 5.2),
        sharex=True,
        sharey=True
    )

    if len(CLASS_LABELS) == 1:
        axes = [axes]

    auc_rows = []

    for ax, c in zip(axes, CLASS_LABELS):
        class_name = CLASS_NAME_MAP.get(c, f"Class {c}")

        for fold_id, subdf in df.groupby("fold"):
            y_true_binary = (subdf["y_true"].values.astype(int) == c).astype(int)
            y_score = subdf[f"prob_class_{c}"].values.astype(float)

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc_i = roc_auc_score(y_true_binary, y_score)

            ax.plot(
                fpr,
                tpr,
                lw=1.8,
                alpha=0.75,
                label=f"Fold {int(fold_id)} AUC={auc_i:.3f}"
            )

            auc_rows.append({
                "class": c,
                "class_name": class_name,
                "fold": int(fold_id),
                "auc": auc_i
            })

        # pooled OOF curve for this class
        y_true_binary_all = (df["y_true"].values.astype(int) == c).astype(int)
        y_score_all = df[f"prob_class_{c}"].values.astype(float)

        fpr_all, tpr_all, _ = roc_curve(y_true_binary_all, y_score_all)
        auc_all = roc_auc_score(y_true_binary_all, y_score_all)

        ax.plot(
            fpr_all,
            tpr_all,
            color="black",
            linestyle="--",
            lw=2.8,
            label=f"OOF AUC={auc_all:.3f}"
        )

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

        ax.set_title(f"{class_name} vs Rest", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", fontsize=8)

        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")

    axes[0].set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")

    if model_name == "all_features":
        fig_title = "Fold-wise Class-specific ROC Curves of All-feature XGBoost"
    elif model_name == "shap_union_top170_180":
        fig_title = "Fold-wise Class-specific ROC Curves of SHAP-selected XGBoost"
    else:
        fig_title = f"Fold-wise Class-specific ROC Curves of {model_name}"

    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.03)

    plt.tight_layout()

    out_prefix = plot_dir / f"{model_name}_foldwise_class_roc"

    plt.savefig(str(out_prefix) + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close()

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(
        plot_dir / f"{model_name}_foldwise_class_auc.csv",
        index=False
    )

    print(f"Saved: {out_prefix}.png")


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        model_dir = RESULT_DIR / model_name

        df_pred = load_fold_predictions(
            model_dir=model_dir,
            model_name=model_name
        )

        plot_foldwise_macro_roc(
            model_dir=model_dir,
            model_name=model_name,
            df=df_pred
        )

        plot_foldwise_class_roc(
            model_dir=model_dir,
            model_name=model_name,
            df=df_pred
        )
    
    # %%