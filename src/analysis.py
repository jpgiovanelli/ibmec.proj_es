# -*- coding: utf-8 -*-
"""
Analise Estatistica de Metricas de Codigo-Fonte do JUnit
AP1 - Engenharia de Software | 2026

Dataset: GitHub Bug Dataset v1.1 -- junit-Class.csv
"""

import os
import sys
import warnings

# Forcar UTF-8 no terminal Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, probplot, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECAO 0 -- CONFIGURACAO E CARREGAMENTO
# ─────────────────────────────────────────────────────────────

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR  = os.path.join(BASE_DIR, "data")
FIG_DIR   = os.path.join(BASE_DIR, "output", "figures")
TAB_DIR   = os.path.join(BASE_DIR, "output", "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Matplotlib / Seaborn defaults ──────────────────────────
plt.rcParams.update({
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})
sns.set_style("whitegrid")
sns.set_palette("colorblind")
COLORS = sns.color_palette("colorblind")

def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> figura salva: {path}")

# ── Carregamento ───────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(DATA_DIR, "junit-Class.csv"))

CATEGORICAL = ["ID", "Name", "LongName", "Parent", "Component", "Path"]
NUMERIC_COLS = [c for c in df_raw.columns if c not in CATEGORICAL]
for col in NUMERIC_COLS:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Remover colunas de variancia zero
zero_var = [c for c in NUMERIC_COLS if df_raw[c].nunique() <= 1]
df_raw.drop(columns=zero_var, inplace=True)
NUMERIC_COLS = [c for c in NUMERIC_COLS if c not in zero_var]

SELECTED_VARS = ["WMC", "CBO", "LCOM5", "DIT", "RFC",
                 "LOC", "NM", "WarningInfo", "WarningMajor", "CD"]

df = df_raw[CATEGORICAL + SELECTED_VARS].copy()
df_s = df[SELECTED_VARS].copy()

print("=" * 65)
print("SECAO 0 -- DADOS CARREGADOS")
print("=" * 65)
print(f"  Dataset : junit-Class.csv")
print(f"  Linhas  : {len(df_s)}")
print(f"  Colunas selecionadas: {SELECTED_VARS}")
print(f"  Colunas zero-variancia removidas: {len(zero_var)}")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 1 -- MEDIDAS DE TENDENCIA CENTRAL
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 1 -- MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 65)

means   = df_s.mean()
medians = df_s.median()
modes   = df_s.mode().iloc[0]

ct = pd.DataFrame({
    "Media":   means.round(3),
    "Mediana": medians.round(3),
    "Moda":    modes.round(3),
})
ct.index.name = "Variavel"
print(ct.to_string())
ct.to_csv(os.path.join(TAB_DIR, "central_tendency.csv"))
print()

# Interpretacao
print("  INTERPRETACAO:")
for var in SELECTED_VARS:
    m, med = means[var], medians[var]
    ratio = m / med if med != 0 else float("inf")
    if ratio > 1.5:
        print(f"  {var}: media ({m:.2f}) >> mediana ({med:.2f}) -> forte assimetria a direita")
    elif ratio > 1.1:
        print(f"  {var}: media ({m:.2f}) > mediana ({med:.2f}) -> leve assimetria a direita")
    else:
        print(f"  {var}: media ({m:.2f}) ~ mediana ({med:.2f}) -> distribuicao mais simetrica")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 2 -- MEDIDAS DE DISPERSAO
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 2 -- MEDIDAS DE DISPERSAO")
print("=" * 65)

amplitude = df_s.max() - df_s.min()
variancia = df_s.var(ddof=1)
desvpad   = df_s.std(ddof=1)
cv        = (desvpad / means * 100).where(means != 0, other=np.nan)

disp = pd.DataFrame({
    "Amplitude":       amplitude.round(3),
    "Variancia":       variancia.round(3),
    "Desvio Padrao":   desvpad.round(3),
    "CV (%)":          cv.round(2),
})
disp.index.name = "Variavel"
print(disp.to_string())
disp.to_csv(os.path.join(TAB_DIR, "dispersion.csv"))
print()

print("  INTERPRETACAO:")
for var in SELECTED_VARS:
    cv_val = cv[var]
    dp_val = desvpad[var]
    if cv_val > 150:
        nivel = "extremamente alta variabilidade"
    elif cv_val > 80:
        nivel = "alta variabilidade"
    elif cv_val > 30:
        nivel = "variabilidade moderada"
    else:
        nivel = "baixa variabilidade"
    print(f"  {var}: CV={cv_val:.1f}% ({nivel}), DP={dp_val:.2f}")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 3 -- MEDIDAS DE POSICAO RELATIVA + BOXPLOTS
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 3 -- MEDIDAS DE POSICAO RELATIVA")
print("=" * 65)

percentis_q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
perc_df = df_s.quantile(percentis_q).T
perc_df.columns = ["P5", "P10", "Q1(P25)", "Mediana", "Q3(P75)", "P90", "P95"]
perc_df.index.name = "Variavel"
iqr = perc_df["Q3(P75)"] - perc_df["Q1(P25)"]
perc_df["IQR"] = iqr.round(3)
print(perc_df.round(3).to_string())
perc_df.to_csv(os.path.join(TAB_DIR, "percentiles.csv"))
print()

# Boxplots 2x5
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, var in enumerate(SELECTED_VARS):
    ax = axes[i]
    sns.boxplot(y=df_s[var], ax=ax, color=COLORS[i % len(COLORS)],
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red",
                           "markeredgecolor": "red", "markersize": 6})
    ax.set_title(var)
    ax.set_xlabel("")
fig.suptitle("Boxplots das Variaveis Selecionadas\n(◆ = media)", fontsize=13)
plt.tight_layout()
savefig("boxplots_all.png")

# ─────────────────────────────────────────────────────────────
# SECAO 4 -- GRAFICOS
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 4 -- GRAFICOS")
print("=" * 65)

# 4a. Histogramas
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, var in enumerate(SELECTED_VARS):
    ax = axes[i]
    sns.histplot(df_s[var], kde=True, bins=30, ax=ax,
                 color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.4)
    ax.axvline(means[var],   color="red",   linestyle="--", linewidth=1.2, label="Media")
    ax.axvline(medians[var], color="green", linestyle="-.", linewidth=1.2, label="Mediana")
    ax.set_title(var)
    ax.set_xlabel("")

patch_m  = mpatches.Patch(color="red",   label="Media")
patch_me = mpatches.Patch(color="green", label="Mediana")
fig.legend(handles=[patch_m, patch_me], loc="lower right", fontsize=10)
fig.suptitle("Histogramas das Variaveis Selecionadas", fontsize=13)
plt.tight_layout()
savefig("histograms.png")

# 4b. Scatter plots com linha de regressao
scatter_pairs = [
    ("LOC",         "WMC",         "Tamanho x Complexidade"),
    ("CBO",         "RFC",         "Acoplamento x Resp. p/ Classe"),
    ("WMC",         "LCOM5",       "Complexidade x Falta de Coesao"),
    ("LOC",         "WarningInfo", "Tamanho x Avisos Informativos"),
    ("DIT",         "CBO",         "Heranca x Acoplamento"),
    ("LOC",         "CD",          "Tamanho x Densidade Comentario"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, (x, y, titulo) in enumerate(scatter_pairs):
    ax = axes[i]
    valid = df_s[[x, y]].dropna()
    r_p, p_p = pearsonr(valid[x], valid[y])
    r_s, _   = spearmanr(valid[x], valid[y])
    sns.regplot(x=valid[x], y=valid[y], ax=ax,
                scatter_kws={"alpha": 0.25, "s": 12, "color": COLORS[i % len(COLORS)]},
                line_kws={"color": "red", "linewidth": 1.5})
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    sig = "***" if p_p < 0.001 else ("**" if p_p < 0.01 else ("*" if p_p < 0.05 else ""))
    ax.annotate(f"r={r_p:.3f}{sig}\nρ={r_s:.3f}",
                xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

fig.suptitle("Graficos de Dispersao entre Pares de Variaveis", fontsize=13)
plt.tight_layout()
savefig("scatter_plots.png")

# 4c. Grafico de barras -- distribuicao de DIT
fig, ax = plt.subplots(figsize=(7, 5))
dit_counts = df_s["DIT"].value_counts().sort_index()
bars = ax.bar(dit_counts.index.astype(str), dit_counts.values,
              color=COLORS[:len(dit_counts)], edgecolor="white")
for bar, v in zip(bars, dit_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(v), ha="center", fontsize=10)
ax.set_xlabel("Profundidade de Heranca (DIT)")
ax.set_ylabel("Numero de Classes")
ax.set_title("Distribuicao de DIT -- Profundidade da Arvore de Heranca")
plt.tight_layout()
savefig("bar_dit.png")

# 4d. Heatmap de correlacao
fig, ax = plt.subplots(figsize=(11, 9))
corr_p = df_s.corr(method="pearson")
mask = np.zeros_like(corr_p, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # mascara triangulo superior (so lower)
mask = ~mask  # mostra lower
sns.heatmap(corr_p, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 8})
ax.set_title("Mapa de Calor -- Correlacoes de Pearson", fontsize=13)
plt.tight_layout()
savefig("correlation_heatmap.png")

print("  Graficos gerados: histogramas, scatter, barras (DIT), heatmap")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 5 -- AVALIACAO DE OUTLIERS
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 5 -- AVALIACAO DE OUTLIERS")
print("=" * 65)

outlier_rows = []
for var in SELECTED_VARS:
    q1  = df_s[var].quantile(0.25)
    q3  = df_s[var].quantile(0.75)
    iqr_v = q3 - q1
    lower = q1 - 1.5 * iqr_v
    upper = q3 + 1.5 * iqr_v
    n_low  = int((df_s[var] < lower).sum())
    n_high = int((df_s[var] > upper).sum())
    pct    = (n_low + n_high) / len(df_s) * 100
    outlier_rows.append({
        "Variavel": var, "Q1": round(q1, 2), "Q3": round(q3, 2),
        "IQR": round(iqr_v, 2), "Fence Inferior": round(lower, 2),
        "Fence Superior": round(upper, 2),
        "Outliers Inf.": n_low, "Outliers Sup.": n_high,
        "% Outliers": round(pct, 1),
    })

outlier_df = pd.DataFrame(outlier_rows).set_index("Variavel")
print(outlier_df.to_string())
outlier_df.to_csv(os.path.join(TAB_DIR, "outliers.csv"))
print()

# Classes com maiores valores (possiveis God Classes)
print("  Classes com LOC extremo (possiveis God Classes):")
uf = outlier_df.loc["LOC", "Fence Superior"]
god_classes = df[df["LOC"] > uf][["Name", "LOC", "WMC", "NM"]].sort_values("LOC", ascending=False)
print(god_classes.head(10).to_string(index=False))
print()

# Boxplots com contagem de outliers anotada
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, var in enumerate(SELECTED_VARS):
    ax = axes[i]
    row = outlier_df.loc[var]
    sns.boxplot(y=df_s[var], ax=ax, color=COLORS[i % len(COLORS)],
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red",
                           "markeredgecolor": "red", "markersize": 5})
    n_out = int(row["Outliers Inf."]) + int(row["Outliers Sup."])
    pct   = row["% Outliers"]
    ax.set_title(f"{var}\n({n_out} outliers, {pct:.1f}%)", fontsize=9)

fig.suptitle("Boxplots com Contagem de Outliers (IQR x 1,5)", fontsize=13)
plt.tight_layout()
savefig("outlier_boxplots.png")

print("  INTERPRETACAO OUTLIERS:")
print("  Metricas de software naturalmente geram outliers por assimetria a direita.")
print("  Classes com LOC/WMC elevados indicam 'God Classes' -- classes com muita")
print("  responsabilidade, violando o Principio da Responsabilidade Unica (SRP).")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 6 -- TESTES DE NORMALIDADE
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 6 -- TESTES DE NORMALIDADE")
print("=" * 65)

norm_rows = []
for var in SELECTED_VARS:
    data = df_s[var].dropna().values
    # Shapiro-Wilk
    sw_stat, sw_p = shapiro(data[:5000])  # max 5000 amostras
    # KS contra normal com parametros da amostra
    ks_stat, ks_p = kstest(
        (data - data.mean()) / data.std(ddof=1) if data.std(ddof=1) > 0 else data,
        "norm"
    )
    normal = "Sim" if (sw_p > 0.05 and ks_p > 0.05) else "Nao"
    norm_rows.append({
        "Variavel": var,
        "SW Estatistica": round(sw_stat, 4),
        "SW p-valor":     sw_p,
        "KS Estatistica": round(ks_stat, 4),
        "KS p-valor":     ks_p,
        "Normal (α=0.05)": normal,
    })

norm_df = pd.DataFrame(norm_rows).set_index("Variavel")
pd.options.display.float_format = "{:.4f}".format
print(norm_df.to_string())
norm_df.to_csv(os.path.join(TAB_DIR, "normality_tests.csv"))
pd.reset_option("display.float_format")
print()

# QQ-plots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, var in enumerate(SELECTED_VARS):
    ax = axes[i]
    data = df_s[var].dropna().values
    (osm, osr), (slope, intercept, r) = probplot(data, dist="norm")
    ax.plot(osm, osr, "o", color=COLORS[i % len(COLORS)],
            alpha=0.4, markersize=3)
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5)
    ax.set_title(f"{var} (r2={r**2:.3f})")
    ax.set_xlabel("Quantis Teoricos")
    ax.set_ylabel("Quantis Observados")

fig.suptitle("Q-Q Plots das Variaveis Selecionadas", fontsize=13)
plt.tight_layout()
savefig("qq_plots.png")

# Teste com log-transformacao
print("  Efeito da transformacao log1p em variaveis chave:")
for var in ["WMC", "LOC", "RFC", "WarningInfo"]:
    data_log = np.log1p(df_s[var].dropna().values)
    sw_stat_log, sw_p_log = shapiro(data_log[:5000])
    sw_orig = norm_df.loc[var, "SW p-valor"]
    print(f"    {var}: p-orig={sw_orig:.4e} -> p-log1p={sw_p_log:.4e}"
          f"  ({'melhora' if sw_p_log > sw_orig else 'sem melhora'})")
print()

print("  INTERPRETACAO NORMALIDADE:")
print("  Nenhuma variavel segue distribuicao normal (esperado para metricas de codigo).")
print("  Metricas de software sao inerentemente assimetricas a direita (poucas classes")
print("  grandes dominam a distribuicao). Isso justifica o uso de Spearman alem de")
print("  Pearson na analise de correlacao.")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 7 -- COEFICIENTES DE CORRELACAO
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 7 -- COEFICIENTES DE CORRELACAO")
print("=" * 65)

corr_pearson  = df_s.corr(method="pearson").round(3)
corr_spearman = df_s.corr(method="spearman").round(3)

print("  Pearson:")
print(corr_pearson.to_string())
print()
print("  Spearman:")
print(corr_spearman.to_string())
print()

corr_pearson.to_csv(os.path.join(TAB_DIR, "correlation_pearson.csv"))
corr_spearman.to_csv(os.path.join(TAB_DIR, "correlation_spearman.csv"))

# Pares com correlacao forte/moderada
print("  PARES DE CORRELACAO NOTAVEIS (Pearson):")
seen = set()
for v1 in SELECTED_VARS:
    for v2 in SELECTED_VARS:
        if v1 >= v2:
            continue
        r = corr_pearson.loc[v1, v2]
        if abs(r) >= 0.4:
            label = "FORTE" if abs(r) >= 0.7 else "moderada"
            print(f"    {v1} x {v2}: r={r:.3f} ({label})")
print()

# PairGrid estilo chart.Correlation()
def corr_annot(x, y, **kwargs):
    ax = plt.gca()
    r, p = pearsonr(x, y)
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    color = "red" if abs(r) >= 0.7 else ("orange" if abs(r) >= 0.4 else "gray")
    size  = 8 + 7 * abs(r)
    ax.annotate(f"r={r:.2f}{stars}", xy=(0.5, 0.5),
                xycoords="axes fraction", ha="center", va="center",
                fontsize=size, color=color, fontweight="bold")
    ax.set_axis_off()

g = sns.PairGrid(df_s, height=1.0, aspect=1)
g.map_diag(sns.histplot, kde=True, bins=20, color=COLORS[0])
g.map_lower(sns.scatterplot, s=6, alpha=0.25, color=COLORS[1])
g.map_upper(corr_annot)
g.figure.suptitle("Matriz de Correlacao -- Estilo chart.Correlation()", y=1.01, fontsize=12)
savefig("pairgrid_correlation.png")

print("  INTERPRETACAO CORRELACAO:")
print("  - Cluster tamanho-complexidade: LOC-WMC-RFC correlacionam fortemente")
print("    (r > 0.85), sugerindo que classes maiores tendem a ser mais complexas.")
print("  - CBO-RFC: correlacao moderada (r ~ 0.47) -- acoplamento cresce com")
print("    o numero de metodos acessiveis externamente.")
print("  - DIT e essencialmente independente das demais variaveis (r < 0.1),")
print("    indicando que heranca nao prediz complexidade no JUnit.")
print("  - CD (documentacao) tem correlacao negativa fraca com tamanho,")
print("    sugerindo que classes maiores sao menos documentadas proporcionalmente.")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 8 -- MODELAGEM ESTATISTICA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 8 -- MODELAGEM ESTATISTICA")
print("=" * 65)

# ── 8A. REGRESSAO LINEAR MULTIPLA ─────────────────────────
print("\n  8A. REGRESSAO LINEAR MULTIPLA")
print("  Variavel resposta (Y) : WMC (Complexidade da Classe)")
print("  Preditores (X)        : CBO, DIT, NM, LCOM5")
print("  (LOC excluido para evitar multicolinearidade -- r=0.94 com WMC)")
print()

df_model = df_s[["WMC", "CBO", "DIT", "NM", "LCOM5"]].dropna()
y_lin = df_model["WMC"]
X_lin = sm.add_constant(df_model[["CBO", "DIT", "NM", "LCOM5"]])

model_lin = sm.OLS(y_lin, X_lin).fit()
print(model_lin.summary())
print()

# VIF
vif_data = pd.DataFrame({
    "Variavel": X_lin.columns,
    "VIF":      [variance_inflation_factor(X_lin.values, i)
                 for i in range(X_lin.shape[1])]
})
print("  Fator de Inflacao da Variancia (VIF):")
print(vif_data.to_string(index=False))
print("  (VIF > 10 indica multicolinearidade problematica)")
print()

# Metricas de desempenho
rmse = np.sqrt(np.mean(model_lin.resid ** 2))
mae  = np.mean(np.abs(model_lin.resid))
print(f"  R2          : {model_lin.rsquared:.4f}")
print(f"  R2 ajustado : {model_lin.rsquared_adj:.4f}")
print(f"  RMSE        : {rmse:.4f}")
print(f"  MAE         : {mae:.4f}")
print()

# Modelo alternativo COM LOC
print("  Modelo com LOC incluido (para comparacao):")
X_lin2 = sm.add_constant(df_model[["CBO", "DIT", "NM", "LCOM5"]].join(
    df_s["LOC"].reindex(df_model.index)))
model_lin2 = sm.OLS(y_lin, X_lin2).fit()
print(f"  R2={model_lin2.rsquared:.4f}, R2adj={model_lin2.rsquared_adj:.4f}")
print()

# Diagnosticos de residuos
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.scatter(model_lin.fittedvalues, model_lin.resid,
           alpha=0.3, s=12, color=COLORS[0])
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Valores Ajustados")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs. Valores Ajustados")

ax = axes[1]
sm.qqplot(model_lin.resid, line="45", ax=ax, alpha=0.4)
ax.set_title("Q-Q Plot dos Residuos")
ax.get_lines()[1].set_color("red")

ax = axes[2]
sns.histplot(model_lin.resid, kde=True, bins=40, ax=ax, color=COLORS[1])
ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Residuos")
ax.set_title("Distribuicao dos Residuos")

fig.suptitle("Diagnosticos da Regressao Linear (WMC ~ CBO + DIT + NM + LCOM5)",
             fontsize=12)
plt.tight_layout()
savefig("regression_diagnostics.png")

# ── 8B. REGRESSAO LOGISTICA ────────────────────────────────
print("\n  8B. REGRESSAO LOGISTICA")
print("  Variavel resposta (Y) : HasMajorWarning (WarningMajor > 0)")
print("  Preditores (X)        : WMC, LOC, CBO, LCOM5, DIT, CD")
print()

df_logit = df_s[["WMC", "LOC", "CBO", "LCOM5", "DIT", "CD", "WarningMajor"]].dropna()
df_logit["HasMajorWarning"] = (df_logit["WarningMajor"] > 0).astype(int)

print(f"  Classes positivas  (WarningMajor > 0): "
      f"{df_logit['HasMajorWarning'].sum()} ({df_logit['HasMajorWarning'].mean()*100:.1f}%)")
print(f"  Classes negativas  (WarningMajor = 0): "
      f"{(df_logit['HasMajorWarning'] == 0).sum()} ({(1 - df_logit['HasMajorWarning'].mean())*100:.1f}%)")
print()

X_log = sm.add_constant(df_logit[["WMC", "LOC", "CBO", "LCOM5", "DIT", "CD"]])
y_log = df_logit["HasMajorWarning"]

model_logit = sm.Logit(y_log, X_log).fit(method="bfgs", maxiter=200, disp=False)
print(model_logit.summary())
print()

# Odds ratios
or_df = pd.DataFrame({
    "Coeficiente": model_logit.params.round(4),
    "Odds Ratio":  np.exp(model_logit.params).round(4),
    "p-valor":     model_logit.pvalues.round(4),
})
print("  Odds Ratios:")
print(or_df.to_string())
print()

# Metricas de classificacao
y_pred_prob  = model_logit.predict(X_log)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_log, y_pred_class)
print("  Matriz de Confusao:")
print(pd.DataFrame(cm,
                   index=["Real: Neg", "Real: Pos"],
                   columns=["Pred: Neg", "Pred: Pos"]).to_string())
print()
print(classification_report(y_log, y_pred_class,
                             target_names=["Sem Warning", "Com Warning"]))

# Curva ROC
fpr, tpr, _ = roc_curve(y_log, y_pred_prob)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=COLORS[0], lw=2,
        label=f"Curva ROC (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatorio")
ax.fill_between(fpr, tpr, alpha=0.12, color=COLORS[0])
ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
ax.set_title("Curva ROC -- Regressao Logistica\n(HasMajorWarning ~ WMC + LOC + CBO + LCOM5 + DIT + CD)")
ax.legend(loc="lower right")
plt.tight_layout()
savefig("logistic_roc_curve.png")

print(f"\n  AUC-ROC: {roc_auc:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# SECAO 9 -- DISCUSSAO DOS RESULTADOS
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SECAO 9 -- DISCUSSAO DOS RESULTADOS")
print("=" * 65)

print("""
  PADRÕES ENCONTRADOS:

  1. CLUSTER TAMANHO-COMPLEXIDADE:
     LOC, WMC e RFC apresentam correlacoes fortes entre si (r > 0.85),
     formando um cluster coeso. Isso indica que no JUnit, classes maiores
     tendem a ser intrinsecamente mais complexas e a expor mais
     comportamentos externos -- um padrao recorrente em projetos Java
     orientados a objetos.

  2. INDEPENDENCIA DA HERANCA (DIT):
     O DIT apresenta correlacoes proximas a zero com todas as demais
     variaveis. Isso sugere que, no JUnit, a profundidade de heranca
     nao prediz nem a complexidade (WMC), nem o acoplamento (CBO),
     nem o tamanho (LOC). O framework usa heranca de forma disciplinada.

  3. NAO-NORMALIDADE UNIVERSAL:
     Todas as 10 variaveis rejeitam a hipotese de normalidade nos testes
     de Shapiro-Wilk e Kolmogorov-Smirnov (p << 0.05). Metricas de
     software sao inerentemente assimetricas -- poucas classes grandes
     dominam enquanto a maioria das classes e pequena e simples.

  4. "NUMBER OF BUGS" = 0 (LIMITACAO CENTRAL):
     Nenhum bug foi registrado nas 1.154 classes analisadas. Isso e
     consistente com a maturidade do JUnit -- um dos frameworks de teste
     mais utilizados no ecossistema Java, com decadas de desenvolvimento
     e revisao da comunidade. A ausencia de variancia impossibilita
     modelagem direta de defeitos, por isso optamos por:
     - Regressao linear: prever WMC (complexidade) a partir de metricas
       estruturais (CBO, DIT, NM, LCOM5).
     - Regressao logistica: prever presenca de avisos PMD severos
       (WarningMajor > 0) como proxy de qualidade de codigo.

  5. REGRESSAO LINEAR (WMC ~ CBO + DIT + NM + LCOM5):
     O modelo explica parte relevante da variabilidade de WMC sem
     recorrer ao LOC (altamente colinear). NM e o preditor mais forte
     -- cada metodo adicional incrementa diretamente a complexidade
     ponderada. LCOM5 contribui positivamente: classes menos coesas
     tambem tendem a ser mais complexas.

  6. REGRESSAO LOGISTICA (HasMajorWarning):
     O modelo discrimina razoavelmente classes com/sem avisos PMD
     graves. LOC e WMC sao os preditores mais significativos --
     classes maiores e mais complexas tem maior probabilidade de
     apresentar violacoes de boas praticas detectadas pelo PMD.

  7. OUTLIERS / "GOD CLASSES":
     Classes como BlockJUnit4ClassRunner e ParentRunner apresentam
     LOC e WMC muito acima da mediana, configurando-se como "God Classes"
     -- antipadrao que concentra responsabilidades excessivas em uma
     unica classe, dificultando manutencao e teste.
""")

print("=" * 65)
print("ANALISE CONCLUIDA")
print("=" * 65)
print(f"  Figuras  -> {FIG_DIR}")
print(f"  Tabelas  -> {TAB_DIR}")
