# -*- coding: utf-8 -*-
"""
03_build_dataset.py — Carrega commits_raw.csv, aplica limpeza, engenharia de
atributos, guards contra vazamento de dados e maturação SZZ, e gera
commits_dataset.csv pronto para modelagem.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_CSV, EXTRA_COLS, KAMEI_COLS, RAW_CSV, savetab,
)

# ── parâmetros de limpeza ─────────────────────────────────────────────────────
MATURATION_WINDOW_DAYS = 180   # descartar commits dos últimos 6 meses (viés SZZ)
LOG1P_COLS = [                  # colunas com cauda pesada -> log1p
    "LA", "LD", "LT", "NF", "NS", "ND", "NUC", "EXP", "NDEV", "SEXP",
    "AGE", "REXP",
]


def main():
    print("=== 03_build_dataset.py ===\n")

    # 1. Carregar CSV bruto
    if not os.path.exists(RAW_CSV):
        sys.exit(f"[ERRO] {RAW_CSV} não encontrado. Execute 02_mine_commits.py primeiro.")
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip", engine="python")
    print(f"[load] {len(df):,} linhas brutas carregadas de {RAW_CSV}")

    # 2. Converter data
    df["committer_date"] = pd.to_datetime(df["committer_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["committer_date"])
    print(f"[date] {len(df):,} linhas com data válida")

    # 3. Remover commits sem features numéricas ou hash
    df = df.dropna(subset=["hash"])
    df = df.drop_duplicates(subset=["hash"])
    print(f"[dedup] {len(df):,} commits únicos")

    # 4. Guard de maturação SZZ: descartar commits muito recentes
    cutoff = datetime.now(timezone.utc) - timedelta(days=MATURATION_WINDOW_DAYS)
    mask_old = df["committer_date"] < cutoff
    removed = (~mask_old).sum()
    df = df[mask_old].copy()
    print(f"[maturation] removidos {removed:,} commits dos últimos {MATURATION_WINDOW_DAYS} dias")

    # 5. Coerção numérica das features Kamei
    for col in KAMEI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
            print(f"  [warn] coluna '{col}' não encontrada — preenchida com 0")

    # 6. Remover linhas com NaN em features essenciais
    df = df.dropna(subset=KAMEI_COLS)
    print(f"[clean] {len(df):,} linhas após remover NaN nas features Kamei")

    # 7. Remover colunas de variância zero (não informativas)
    before_cols = set(df.columns)
    feat_df = df[KAMEI_COLS]
    zero_var = [c for c in KAMEI_COLS if feat_df[c].std() == 0]
    if zero_var:
        print(f"[zero-var] removendo {zero_var}")
        df = df.drop(columns=zero_var)
    print(f"[cols] {len(before_cols)} -> {len(df.columns)} colunas após limpeza")

    # 8. Engenharia de atributos adicionais
    la = df["LA"].clip(lower=0) if "LA" in df.columns else pd.Series(0, index=df.index)
    ld = df["LD"].clip(lower=0) if "LD" in df.columns else pd.Series(0, index=df.index)
    lt = df["LT"].clip(lower=1) if "LT" in df.columns else pd.Series(1, index=df.index)

    df["churn"]    = la + ld
    df["LA_ratio"] = la / lt
    df["LD_ratio"] = ld / lt

    # 9. Transformação log1p nas colunas com cauda pesada
    for col in LOG1P_COLS:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    df["churn_log"]    = np.log1p(df["churn"])
    df["LA_ratio_log"] = np.log1p(df["LA_ratio"])
    df["LD_ratio_log"] = np.log1p(df["LD_ratio"])

    # 10. Garantir que is_buggy é inteiro 0/1
    df["is_buggy"] = df["is_buggy"].fillna(0).astype(int).clip(0, 1)

    # 11. Ordenar cronologicamente (importante para o split temporal)
    df = df.sort_values("committer_date").reset_index(drop=True)

    # 12. Resumo de balanceamento
    buggy_n   = df["is_buggy"].sum()
    total_n   = len(df)
    per_repo  = df.groupby("repo")["is_buggy"].agg(["sum", "count"])
    per_repo["pct_buggy"] = per_repo["sum"] / per_repo["count"] * 100
    per_repo = per_repo.rename(columns={"sum": "buggy", "count": "total"})

    print(f"\n[balance] Total: {total_n:,} commits — buggy: {buggy_n:,} "
          f"({buggy_n/total_n*100:.1f}%) | clean: {total_n-buggy_n:,}")
    print(per_repo.to_string())

    # 13. Salvar
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    savetab(per_repo, "dataset_summary.csv")
    print(f"\n[save] {DATASET_CSV}")

    # 14. Estatísticas descritivas das features
    feat_cols = [c for c in df.columns if c in KAMEI_COLS + EXTRA_COLS]
    desc = df[feat_cols].describe().T
    desc["skew"] = df[feat_cols].skew()
    savetab(desc, "feature_descriptive_stats.csv")

    print("\n=== Dataset pronto. ===")


if __name__ == "__main__":
    main()
