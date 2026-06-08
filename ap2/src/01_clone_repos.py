# -*- coding: utf-8 -*-
"""
01_clone_repos.py — Clona (ou atualiza) os repositórios listados em config.py.
Idempotente: pula repos já clonados. Exibe tamanho estimado de cada clone.
"""

import os
import subprocess
import sys

# adiciona src/ ao path para importar config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPO_DIR, REPOS


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1_048_576


def clone_or_skip(name: str, url: str) -> None:
    dest = os.path.join(REPO_DIR, name)
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"  [skip] {name} — já existe em {dest}")
        return
    print(f"  [clone] {name} <- {url} ...")
    subprocess.run(
        ["git", "clone", "--no-tags", "--filter=blob:none", url, dest],
        check=True,
    )
    size = dir_size_mb(dest)
    print(f"  [done]  {name} ({size:.1f} MB)")


def main():
    print(f"=== Clonando {len(REPOS)} repositórios em {REPO_DIR} ===\n")
    for name, url in REPOS:
        try:
            clone_or_skip(name, url)
        except subprocess.CalledProcessError as exc:
            print(f"  [ERRO] {name}: {exc}", file=sys.stderr)
    print("\n=== Clone concluído. ===")


if __name__ == "__main__":
    main()
