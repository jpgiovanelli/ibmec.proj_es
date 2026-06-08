# -*- coding: utf-8 -*-
"""
04_model_compare.py — Carrega commits_dataset.csv, executa divisão temporal e
aleatória, treina e compara modelos de ML, avalia com métricas completas,
e gera figuras + tabelas de resultado.

Modelos: Regressão Logística (baseline), Random Forest, XGBoost, LightGBM.
Tarefa de regressão secundária: predição de churn (LA+LD) via OLS + RF.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_CSV, RANDOM_STATE, savetab, savefig, FIG_DIR, TAB_DIR

from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier


# ── features usadas pelo classificador ──────────────────────────────────────
# Variantes log1p (mais estáveis numericamente) + FIX direto
FEATURE_COLS_LOG = [
    "NS_log", "ND_log", "NF_log", "Entropy",
    "LA_log", "LD_log", "LT_log", "FIX",
    "NDEV_log", "AGE_log", "NUC_log",
    "EXP_log", "REXP_log", "SEXP_log",
    "LA_ratio_log", "LD_ratio_log",
]
TARGET_COL  = "is_buggy"
CHURN_COL   = "churn_log"   # alvo da regressão secundária


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["committer_date"] = pd.to_datetime(df["committer_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["committer_date"]).sort_values("committer_date").reset_index(drop=True)
    return df


def available_features(df: pd.DataFrame) -> list[str]:
    """Retorna apenas as colunas de feature que existem no DataFrame."""
    return [c for c in FEATURE_COLS_LOG if c in df.columns]


def temporal_split(df: pd.DataFrame, train_frac: float = 0.75):
    """Split temporal: primeiros train_frac% para treino, restante para teste."""
    n = len(df)
    cutoff = int(n * train_frac)
    train = df.iloc[:cutoff]
    test  = df.iloc[cutoff:]
    return train, test


def evaluate_clf(name: str, model, X_test, y_test) -> dict:
    """Avalia um classificador e retorna dict de métricas."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model":      name,
        "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC":    round(roc_auc_score(y_test, y_proba), 4),
        "PR_AUC":     round(average_precision_score(y_test, y_proba), 4),
    }


def build_models(neg_pos_ratio: float):
    """Retorna dict de (nome -> pipeline) para os classificadores."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300, scale_pos_weight=neg_pos_ratio,
        learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )

    lgbm = LGBMClassifier(
        n_estimators=300, is_unbalance=True,
        learning_rate=0.05, num_leaves=63,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

    return {
        "LogReg":   lr,
        "RandomForest": rf,
        "XGBoost":  xgb,
        "LightGBM": lgbm,
    }


def tune_model(model, X_train, y_train, param_grid: dict, label: str):
    """RandomizedSearchCV com StratifiedKFold(5) e scoring roc_auc."""
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rsc = RandomizedSearchCV(
        model, param_grid, n_iter=20, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    rsc.fit(X_train, y_train)
    print(f"  [{label}] best params: {rsc.best_params_} | CV ROC-AUC: {rsc.best_score_:.4f}")
    return rsc.best_estimator_, rsc.best_params_


# ── hiperparâmetros para tuning ───────────────────────────────────────────────
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators":    [200, 300, 500],
        "max_depth":       [None, 8, 15, 25],
        "min_samples_leaf":[1, 2, 5],
        "max_features":    ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators":    [200, 300, 500],
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators":    [200, 300, 500],
        "num_leaves":      [31, 63, 127],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
    },
}


def plot_roc_curves(models_preds: dict, y_test, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (_, y_proba) in models_preds.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curvas ROC — {suffix}")
    ax.legend(loc="lower right")
    savefig(f"roc_curves{suffix}.png")


def plot_pr_curves(models_preds: dict, y_test, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (_, y_proba) in models_preds.items():
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curvas Precision-Recall — {suffix}")
    ax.legend(loc="upper right")
    savefig(f"pr_curves{suffix}.png")


def plot_confusion_matrices(models_preds: dict, y_test, suffix: str = "") -> None:
    names = list(models_preds.keys())
    n = len(names)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, name in enumerate(names):
        y_pred, _ = models_preds[name]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Buggy"])
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(name)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Matrizes de Confusão — {suffix}", y=1.02, fontsize=13)
    plt.tight_layout()
    savefig(f"confusion_matrices{suffix}.png")


def plot_feature_importance(model, feat_cols: list[str], model_name: str) -> None:
    """Plota importância de features (apenas para modelos baseados em árvores)."""
    try:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feat_ser = pd.Series(importances, index=feat_cols).sort_values(ascending=True)
            top = feat_ser.tail(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            top.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"Importância de Features — {model_name}")
            ax.set_xlabel("Importância (Gini/Ganho)")
            savefig(f"feature_importance_{model_name.lower()}.png")
            # salvar tabela
            savetab(feat_ser.sort_values(ascending=False).to_frame("importance"),
                    f"feature_importance_{model_name.lower()}.csv")
    except Exception as exc:
        print(f"  [warn] feature importance {model_name}: {exc}")


def run_classification(df: pd.DataFrame, feat_cols: list[str], split_type: str):
    """Executa pipeline completo de classificação para um tipo de split."""
    print(f"\n{'='*60}")
    print(f"CLASSIFICAÇÃO — Split: {split_type}")
    print(f"{'='*60}")

    X = df[feat_cols].values
    y = df[TARGET_COL].values

    if split_type == "temporal":
        train_df, test_df = temporal_split(df)
        X_train = train_df[feat_cols].values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[feat_cols].values
        y_test  = test_df[TARGET_COL].values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE,
        )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / max(n_pos, 1)
    print(f"Treino: {len(y_train):,} ({n_pos:,} buggy = {n_pos/len(y_train)*100:.1f}%)")
    print(f"Teste:  {len(y_test):,}  ({y_test.sum():,} buggy = {y_test.sum()/len(y_test)*100:.1f}%)")

    if n_pos == 0:
        print("[ERRO] Nenhum commit buggy no treino — impossível treinar. Verifique a mineração.")
        return None, None, None

    models = build_models(neg_pos_ratio=ratio)

    # tuning para RF, XGB, LGBM
    best_params_all = {}
    for name, param_grid in PARAM_GRIDS.items():
        if name in models:
            print(f"\n[tune] {name}...")
            models[name], bp = tune_model(models[name], X_train, y_train, param_grid, name)
            best_params_all[name] = str(bp)

    # treinar LogReg (sem tuning — baseline simples)
    print("\n[train] LogReg (baseline)...")
    models["LogReg"].fit(X_train, y_train)

    # avaliar todos
    results      = []
    models_preds = {}  # name -> (y_pred, y_proba)
    for name, model in models.items():
        metrics = evaluate_clf(name, model, X_test, y_test)
        results.append(metrics)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        models_preds[name] = (y_pred, y_proba)
        print(f"  {name:15s} F1={metrics['F1']:.4f}  ROC-AUC={metrics['ROC_AUC']:.4f}  "
              f"PR-AUC={metrics['PR_AUC']:.4f}")

    suffix    = "_temporal" if split_type == "temporal" else "_random"
    results_df = pd.DataFrame(results).set_index("Model")
    savetab(results_df, f"model_comparison{suffix}.csv")

    # best params
    params_df = pd.DataFrame(best_params_all.items(), columns=["Model", "Params"]).set_index("Model")
    savetab(params_df, f"best_hyperparams{suffix}.csv")

    # figuras
    plot_roc_curves(models_preds, y_test, suffix)
    plot_pr_curves(models_preds, y_test, suffix)
    plot_confusion_matrices(
        {n: (p, pr) for n, (p, pr) in models_preds.items()},
        y_test, suffix
    )

    # gráfico comparativo de métricas
    metric_cols = ["F1", "ROC_AUC", "PR_AUC", "Recall"]
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df[metric_cols].plot(kind="bar", ax=ax, rot=30)
    ax.set_title(f"Comparação de Modelos — {split_type}")
    ax.set_ylabel("Pontuação")
    ax.legend(loc="lower right")
    savefig(f"model_metrics_bar{suffix}.png")

    # importância de features (árvores)
    for name in ["RandomForest", "XGBoost", "LightGBM"]:
        plot_feature_importance(models[name], feat_cols, name)

    return models, results_df, models_preds


def run_regression(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """Regressão secundária: predição de churn (LA+LD) — reporta RMSE e MAE."""
    print(f"\n{'='*60}")
    print("REGRESSÃO SECUNDÁRIA — Predição de Churn (log(LA+LD+1))")
    print(f"{'='*60}")

    if CHURN_COL not in df.columns:
        print("[skip] coluna churn_log não encontrada.")
        return

    X = df[feat_cols].values
    y = df[CHURN_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE,
    )

    regressors = {
        "Ridge":          Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))]),
        "RandomForestReg": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }

    reg_results = []
    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae  = mean_absolute_error(y_test, y_pred)
        reg_results.append({"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        print(f"  {name:20s} RMSE={rmse:.4f}  MAE={mae:.4f}")

    savetab(pd.DataFrame(reg_results).set_index("Model"), "regression_metrics.csv")


def plot_class_balance(df: pd.DataFrame) -> None:
    per_repo = df.groupby("repo")["is_buggy"].agg(["sum", "count"])
    per_repo.columns = ["buggy", "total"]
    per_repo["clean"] = per_repo["total"] - per_repo["buggy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    per_repo[["clean", "buggy"]].plot(kind="bar", stacked=True, ax=axes[0], color=["steelblue", "tomato"])
    axes[0].set_title("Commits por Repositório")
    axes[0].set_xlabel("Repositório")
    axes[0].set_ylabel("Nº de commits")
    axes[0].tick_params(axis="x", rotation=30)

    overall = pd.Series({"clean": (df["is_buggy"]==0).sum(), "buggy": (df["is_buggy"]==1).sum()})
    overall.plot(kind="bar", ax=axes[1], color=["steelblue", "tomato"])
    axes[1].set_title("Balanceamento geral")
    axes[1].set_ylabel("Nº de commits")
    savefig("class_balance.png")


def plot_feature_distributions(df: pd.DataFrame, feat_cols: list[str]) -> None:
    cols_to_plot = [c for c in feat_cols if c in df.columns][:12]
    n = len(cols_to_plot)
    rows, cols = (n + 3) // 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribuição das Features (log1p)", fontsize=12)
    plt.tight_layout()
    savefig("feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, feat_cols: list[str]) -> None:
    corr = df[feat_cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlação de Spearman entre Features")
    savefig("correlation_heatmap.png")
    savetab(corr, "correlation_matrix.csv")


def plot_time_vs_random(temporal_df: pd.DataFrame, random_df: pd.DataFrame) -> None:
    combined = pd.concat([
        temporal_df.assign(Split="Temporal"),
        random_df.assign(Split="Aleatório"),
    ])
    fig, ax = plt.subplots(figsize=(10, 5))
    combined_pivot = combined.set_index(["Split", combined.index])["F1"].unstack(level=0)
    combined.pivot_table(index="Model", columns="Split", values="F1").plot(kind="bar", ax=ax, rot=30)
    ax.set_title("F1-Score: Split Temporal vs Aleatório (quanto maior a diferença = mais vazamento)")
    ax.set_ylabel("F1-Score")
    savefig("time_vs_random_split.png")


def main():
    print("=== 04_model_compare.py ===\n")

    if not os.path.exists(DATASET_CSV):
        sys.exit(f"[ERRO] {DATASET_CSV} não encontrado. Execute 03_build_dataset.py primeiro.")

    df = load_data(DATASET_CSV)
    print(f"[load] {len(df):,} linhas carregadas | buggy={df['is_buggy'].sum():,} "
          f"({df['is_buggy'].mean()*100:.1f}%)")

    feat_cols = available_features(df)
    print(f"[features] {len(feat_cols)} features disponíveis: {feat_cols}\n")

    if df["is_buggy"].sum() == 0:
        sys.exit("[ERRO] Nenhum commit buggy no dataset — verifique a mineração SZZ.")

    # ── figuras exploratórias ─────────────────────────────────────────────────
    print("[plot] Figuras exploratórias...")
    plot_class_balance(df)
    plot_feature_distributions(df, feat_cols)
    plot_correlation_heatmap(df, feat_cols)

    # ── classificação ─────────────────────────────────────────────────────────
    _, temporal_results, _ = run_classification(df, feat_cols, split_type="temporal")
    _, random_results,   _ = run_classification(df, feat_cols, split_type="random")

    # ── gráfico temporal vs aleatório ─────────────────────────────────────────
    if temporal_results is not None and random_results is not None:
        plot_time_vs_random(temporal_results.reset_index(), random_results.reset_index())

    # ── regressão secundária ──────────────────────────────────────────────────
    run_regression(df, feat_cols)

    print("\n=== Modelagem concluída. Figuras e tabelas salvas. ===")


if __name__ == "__main__":
    main()
