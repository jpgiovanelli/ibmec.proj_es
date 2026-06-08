# -*- coding: utf-8 -*-
"""
02_mine_commits.py — Minera commits dos repositórios com PyDriller e gera
commits_raw.csv. Implementa as 14 métricas Kamei et al. (2013) em uma passada
cronológica única usando dicionários incrementais, e aplica rotulagem SZZ-lite
via get_commits_last_modified_lines().

Uso:
    python 02_mine_commits.py            # processa repos ainda não minerados
    python 02_mine_commits.py --force    # re-minera tudo
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BUG_FIX_RE, KAMEI_COLS, LEDGER_FILE, RAW_CSV, REPO_DIR, REPOS,
)

try:
    from pydriller import Git, Repository
except ImportError:
    sys.exit("PyDriller não encontrado. Execute: pip install pydriller")

# ── constantes ───────────────────────────────────────────────────────────────
PRINT_EVERY   = 200     # progresso a cada N commits
MAX_BLAME_FILES = 8     # limita blame p/ commits que tocam muitos arquivos (perf)
BLANK_OR_CMT    = re.compile(r"^\s*(#.*)?$")  # linhas em branco ou comentário


# ── helpers ──────────────────────────────────────────────────────────────────

def _entropy(counts: list[int]) -> float:
    """Entropia de Shannon normalizada sobre distribuição de mudanças."""
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h / math.log2(len(counts))   # normalizada [0,1]


def _subsystem(path: str) -> str:
    """Retorna o subsistema (diretório raiz) de um caminho."""
    parts = path.replace("\\", "/").split("/")
    return parts[0] if parts else ""


def _is_py(path: str) -> bool:
    return (path or "").endswith(".py")


def _ndays_ago(dt: datetime, ref: datetime) -> float:
    """Retorna diferença em dias entre ref e dt (positivo se dt < ref)."""
    if dt is None or ref is None:
        return 0.0
    try:
        # garante tz-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        delta = (ref - dt).total_seconds() / 86400.0
        return max(delta, 0.0)
    except Exception:
        return 0.0


# ── núcleo da mineração ───────────────────────────────────────────────────────

def mine_repo(name: str, repo_path: str) -> list[dict]:
    """Minera todos os commits de um repositório e retorna lista de dicts."""
    print(f"\n[{name}] Iniciando mineração em {repo_path}")

    # dicionários incrementais (estado acumulado ao longo do tempo)
    file_authors:   dict[str, set]      = defaultdict(set)
    file_last_time: dict[str, datetime] = {}
    file_nchanges:  dict[str, int]      = defaultdict(int)
    auth_commits:   dict[str, int]      = defaultdict(int)
    auth_rexp:      dict[str, float]    = defaultdict(float)
    auth_subsys:    dict[tuple, int]    = defaultdict(int)

    rows: list[dict] = []
    bug_fixing_commits: list = []  # commits de fix para SZZ

    # ── Passagem 1: extrair métricas por commit (ordem cronológica) ──────────
    try:
        repo_iter = Repository(
            repo_path,
            order="date",   # cronológico crescente
        )
    except Exception as exc:
        print(f"  [ERRO] Não foi possível abrir o repo: {exc}", file=sys.stderr)
        return []

    for i, commit in enumerate(repo_iter.traverse_commits()):
        try:
            # pular merges (SZZ em merges é ruidoso)
            if commit.merge:
                continue

            hash_    = commit.hash
            author   = commit.author.email or commit.author.name or "unknown"
            msg      = commit.msg or ""
            cdate    = commit.committer_date

            modified_files = commit.modified_files

            # ── métricas diretas ─────────────────────────────────────────────
            files_changed = [
                mf for mf in modified_files
                if mf.new_path or mf.old_path
            ]
            if not files_changed:
                continue

            paths = [
                (mf.new_path or mf.old_path).replace("\\", "/")
                for mf in files_changed
            ]

            # NS — subsistemas únicos
            subsystems = {_subsystem(p) for p in paths}
            NS = len(subsystems)

            # ND — diretórios únicos
            dirs = {os.path.dirname(p) for p in paths}
            ND = len(dirs)

            # NF — arquivos modificados
            NF = len(files_changed)

            # LA, LD por arquivo
            la_per_file = [mf.added_lines   for mf in files_changed]
            ld_per_file = [mf.deleted_lines  for mf in files_changed]
            LA = sum(la_per_file)
            LD = sum(ld_per_file)

            # LT — média de LOC antes por arquivo
            lt_vals = []
            for mf in files_changed:
                n = mf.nloc
                if n is not None and n >= 0:
                    lt_vals.append(n)
            LT = (sum(lt_vals) / len(lt_vals)) if lt_vals else 0.0

            # Entropy — distribuição das mudanças
            churn_per_file = [a + d for a, d in zip(la_per_file, ld_per_file)]
            Entropy = _entropy(churn_per_file)

            # FIX — é commit de correção?
            FIX = int(bool(BUG_FIX_RE.search(msg)))

            # ── métricas históricas (calculadas ANTES de atualizar estado) ────

            # NDEV — nº de autores distintos que já tocaram os arquivos
            ndev_vals = [len(file_authors[p]) for p in paths]
            NDEV = (sum(ndev_vals) / len(ndev_vals)) if ndev_vals else 0.0

            # AGE — média de dias desde última mudança nos arquivos
            age_vals = [
                _ndays_ago(file_last_time[p], cdate)
                for p in paths if p in file_last_time
            ]
            AGE = (sum(age_vals) / len(age_vals)) if age_vals else 0.0

            # NUC — nº médio de commits anteriores nos arquivos
            nuc_vals = [file_nchanges[p] for p in paths]
            NUC = (sum(nuc_vals) / len(nuc_vals)) if nuc_vals else 0.0

            # EXP — total de commits do autor
            EXP = auth_commits[author]

            # REXP — experiência recente ponderada (soma 1/(idade_em_anos+1))
            REXP = auth_rexp[author]

            # SEXP — experiência do autor nos subsistemas tocados
            SEXP = sum(auth_subsys[(author, s)] for s in subsystems)

            # ── atualizar estado incremental ──────────────────────────────────
            for p in paths:
                file_authors[p].add(author)
                file_last_time[p] = cdate
                file_nchanges[p] += 1
            auth_commits[author] += 1
            commit_age_years = _ndays_ago(cdate, datetime.now(timezone.utc)) / 365.25
            auth_rexp[author] += 1.0 / (commit_age_years + 1.0)
            for s in subsystems:
                auth_subsys[(author, s)] += 1

            row = {
                "repo":           name,
                "hash":           hash_,
                "author":         author,
                "committer_date": cdate.isoformat() if cdate else "",
                "msg_head":       " ".join(msg[:120].split()),
                "NS": NS, "ND": ND, "NF": NF, "Entropy": round(Entropy, 6),
                "LA": LA, "LD": LD, "LT": round(LT, 2),
                "FIX": FIX,
                "NDEV": round(NDEV, 2), "AGE": round(AGE, 2),
                "NUC": round(NUC, 2),
                "EXP": EXP, "REXP": round(REXP, 4), "SEXP": SEXP,
                "is_buggy": 0,   # preenchido na Passagem 2
            }
            rows.append(row)

            if FIX:
                bug_fixing_commits.append(commit)

            if (i + 1) % PRINT_EVERY == 0:
                print(f"  [{name}] {i+1} commits processados ({len(rows)} válidos, "
                      f"{len(bug_fixing_commits)} fixes)...")

        except Exception as exc:
            # commit problemático — pula silenciosamente
            print(f"  [warn] commit {getattr(commit,'hash','?')[:8]}: {exc}")
            continue

    print(f"  [{name}] Passagem 1 concluída: {len(rows)} commits válidos, "
          f"{len(bug_fixing_commits)} bug-fixing commits.")

    # ── Passagem 2: SZZ-lite — rotulagem via blame ────────────────────────────
    if bug_fixing_commits:
        print(f"  [{name}] SZZ-lite: analisando {len(bug_fixing_commits)} BFCs...")
        bug_inducing_hashes: set[str] = set()
        gr = Git(repo_path)

        for fix_commit in bug_fixing_commits:
            try:
                # apenas arquivos .py com linhas deletadas
                py_files_with_del = [
                    mf for mf in fix_commit.modified_files
                    if _is_py(mf.new_path or mf.old_path or "")
                    and mf.deleted_lines > 0
                ]

                # limitar para repositórios muito grandes
                if len(py_files_with_del) > MAX_BLAME_FILES:
                    py_files_with_del = py_files_with_del[:MAX_BLAME_FILES]

                for mf in py_files_with_del:
                    # filtrar linhas em branco / comentário
                    deleted = mf.diff_parsed.get("deleted", [])
                    meaningful = [
                        ln for ln, content in deleted
                        if not BLANK_OR_CMT.match(content)
                    ]
                    if not meaningful:
                        continue

                    inducing = gr.get_commits_last_modified_lines(
                        fix_commit, modification=mf
                    )
                    for path_key, hashes in inducing.items():
                        for h in hashes:
                            bug_inducing_hashes.add(h)

            except Exception as exc:
                print(f"  [warn-szz] {fix_commit.hash[:8]}: {exc}")
                continue

        # marcar commits como buggy
        hash_to_idx = {r["hash"]: idx for idx, r in enumerate(rows)}
        labeled = 0
        for h in bug_inducing_hashes:
            if h in hash_to_idx:
                rows[hash_to_idx[h]]["is_buggy"] = 1
                labeled += 1

        buggy_rate = labeled / len(rows) * 100 if rows else 0
        print(f"  [{name}] SZZ concluído: {labeled} commits rotulados como buggy "
              f"({buggy_rate:.1f}% do total).")
    else:
        print(f"  [{name}] Nenhum BFC encontrado — nenhum commit rotulado como buggy.")

    return rows


# ── entrada principal ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Minera commits com PyDriller + SZZ-lite")
    parser.add_argument("--force", action="store_true", help="Re-minera tudo (ignora ledger e CSV existente)")
    args = parser.parse_args()

    # ledger de repos já minerados
    ledger: set[str] = set()
    if not args.force and os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            ledger = {line.strip() for line in f if line.strip()}

    if args.force and os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
        open(LEDGER_FILE, "w").close()
        ledger.clear()
        print("[force] CSV e ledger resetados.")

    fieldnames = ["repo", "hash", "author", "committer_date", "msg_head"] + \
                 ["NS", "ND", "NF", "Entropy", "LA", "LD", "LT", "FIX",
                  "NDEV", "AGE", "NUC", "EXP", "REXP", "SEXP", "is_buggy"]

    csv_exists = os.path.exists(RAW_CSV)
    csv_mode   = "a" if csv_exists else "w"

    total_written = 0

    with open(RAW_CSV, csv_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                lineterminator="\n", extrasaction="ignore")
        if not csv_exists or args.force:
            writer.writeheader()

        for name, _url in REPOS:
            if name in ledger:
                print(f"[skip] {name} já minerado (ledger). Use --force para re-minerar.")
                continue

            repo_path = os.path.join(REPO_DIR, name)
            if not os.path.isdir(repo_path):
                print(f"[ERRO] {name}: diretório não encontrado ({repo_path}). "
                      f"Execute 01_clone_repos.py primeiro.", file=sys.stderr)
                continue

            rows = mine_repo(name, repo_path)
            if rows:
                writer.writerows(rows)
                csvfile.flush()
                total_written += len(rows)

            # registrar no ledger
            with open(LEDGER_FILE, "a", encoding="utf-8") as lf:
                lf.write(name + "\n")

    print(f"\n=== Mineração concluída. {total_written} commits escritos em {RAW_CSV} ===")


if __name__ == "__main__":
    main()
