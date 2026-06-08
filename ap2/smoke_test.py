import sys, os, importlib.util
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

# Carregar config
import ap2.src.config as cfg

# Sobrescrever repos para smoke test
cfg.REPOS = [
    ("requests", "https://github.com/psf/requests"),
    ("click",    "https://github.com/pallets/click"),
]

# Carregar 02_mine_commits.py via importlib (nome começa com numero)
spec = importlib.util.spec_from_file_location(
    "mine_commits",
    os.path.join(src_dir, "02_mine_commits.py")
)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

import csv

fieldnames = ["repo","hash","author","committer_date","msg_head",
              "NS","ND","NF","Entropy","LA","LD","LT","FIX",
              "NDEV","AGE","NUC","EXP","REXP","SEXP","is_buggy"]

RAW = cfg.RAW_CSV
LGR = cfg.LEDGER_FILE

if os.path.exists(RAW): os.remove(RAW)
if os.path.exists(LGR): open(LGR,"w").close()

all_rows = []
for name, _ in cfg.REPOS:
    rp = os.path.join(cfg.REPO_DIR, name)
    rows = mc.mine_repo(name, rp)
    all_rows.extend(rows)
    b = sum(r["is_buggy"] for r in rows)
    pct = b/max(len(rows),1)*100
    print(f"\n>>> {name}: {len(rows)} commits, buggy={b} ({pct:.1f}%)")

with open(RAW,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(all_rows)

b = sum(r["is_buggy"] for r in all_rows)
print(f"\n=== SMOKE OK: {len(all_rows)} commits, buggy={b} ({b/max(len(all_rows),1)*100:.1f}%) ===")

import pandas as pd
df = pd.read_csv(RAW)
print(df[["NS","LA","LD","FIX","EXP","is_buggy"]].describe().round(2))
