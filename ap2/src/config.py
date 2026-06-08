# -*- coding: utf-8 -*-
"""
config.py — Configurações compartilhadas entre todos os scripts do AP2.
Fornece: paths, lista de repositórios, regex de fix, colunas de features,
boilerplate UTF-8/matplotlib e helper savefig().
"""

import io
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # backend sem display (Windows-safe)
import matplotlib.pyplot as plt
import seaborn as sns

# ── forçar UTF-8 no stdout (Windows) ────────────────────────────────────────
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── paths portáveis ──────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)          # ap2/
DATA_DIR = os.path.join(BASE_DIR, "data")
REPO_DIR = os.path.join(DATA_DIR, "repos")
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
TAB_DIR  = os.path.join(BASE_DIR, "output", "tables")

RAW_CSV     = os.path.join(DATA_DIR, "commits_raw.csv")
DATASET_CSV = os.path.join(DATA_DIR, "commits_dataset.csv")
LEDGER_FILE = os.path.join(DATA_DIR, ".mined_repos.txt")

for _d in [DATA_DIR, REPO_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── lista de repositórios a minerar ─────────────────────────────────────────
# (nome, URL) — 12 projetos Python ativos com histórico rico de bug-fixes
REPOS = [
    ("requests",   "https://github.com/psf/requests"),
    ("flask",      "https://github.com/pallets/flask"),
    ("click",      "https://github.com/pallets/click"),
    ("jinja",      "https://github.com/pallets/jinja"),
    ("black",      "https://github.com/psf/black"),
    ("typer",      "https://github.com/tiangolo/typer"),
    ("httpx",      "https://github.com/encode/httpx"),
    ("starlette",  "https://github.com/encode/starlette"),
    ("tornado",    "https://github.com/tornadoweb/tornado"),
    ("pydantic",   "https://github.com/pydantic/pydantic"),
    ("attrs",      "https://github.com/python-attrs/attrs"),
    ("alembic",    "https://github.com/sqlalchemy/alembic"),
]

# ── regex de detecção de commit de correção de bug ──────────────────────────
BUG_FIX_RE = re.compile(
    r"\b(fix(e[ds])?|fixup|bugfix|hotfix|bug(s)?|defect|patch|"
    r"error|fault|fail(ure|ed|s)?|issue|resolve[ds]?|resolving|"
    r"correct(ed|ion|s)?|repair(ed|s)?|workaround|revert)\b"
    r"|"
    r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s*#\d+",
    re.IGNORECASE,
)

# ── colunas das métricas Kamei (features para o ML) ─────────────────────────
KAMEI_COLS = [
    # dimensão difusão
    "NS",       # nº de subsistemas modificados
    "ND",       # nº de diretórios modificados
    "NF",       # nº de arquivos modificados
    "Entropy",  # distribuição das mudanças pelos arquivos
    # dimensão tamanho
    "LA",       # linhas adicionadas
    "LD",       # linhas removidas
    "LT",       # linhas totais antes (media por arquivo)
    # dimensão propósito
    "FIX",      # é um commit de correção? (0/1)
    # dimensão histórico
    "NDEV",     # nº de devs distintos que já tocaram os arquivos
    "AGE",      # média de dias desde última mudança nos arquivos
    "NUC",      # nº de commits anteriores nos arquivos
    # dimensão experiência do desenvolvedor
    "EXP",      # commits totais do autor
    "REXP",     # experiência recente ponderada pelo tempo
    "SEXP",     # experiência do autor no(s) subsistema(s)
]

# features derivadas adicionadas no build_dataset
EXTRA_COLS = ["LA_ratio", "LD_ratio", "churn"]

# ── configuração visual padrão ───────────────────────────────────────────────
RANDOM_STATE = 42
DPI = 300

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def savefig(filename: str) -> None:
    """Salva a figura corrente em FIG_DIR com o nome informado."""
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig] {path}")


def savetab(df, filename: str) -> None:
    """Salva um DataFrame como CSV em TAB_DIR."""
    path = os.path.join(TAB_DIR, filename)
    df.to_csv(path, index=True, encoding="utf-8")
    print(f"  [tab] {path}")
