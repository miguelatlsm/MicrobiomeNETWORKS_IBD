# This script computes mutual information between microbiome features and
# transcript-level variables within each diagnostic group.
# The resulting matrices are used to define host-microbe links.

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import networkx as nx
from networkx.algorithms import bipartite  # kept (not strictly required here)

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed

# Auxiliary functions

def prep_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how="any")
    X = X.loc[:, X.nunique(dropna=True) > 1]  # quitar constantes
    return X

def top_by_variance(X: pd.DataFrame, top_n: int) -> pd.DataFrame:
    v = X.var(axis=0).sort_values(ascending=False)
    return X.loc[:, v.index[:top_n]]

def collapse_duplicate_columns_mean(X: pd.DataFrame) -> pd.DataFrame:
    return X.T.groupby(level=0).mean().T

def get_X_by_group(df_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    micro_cols = [c for c in df_sub.columns if str(c).startswith("micro__")]
    tx_cols = [c for c in df_sub.columns if str(c).startswith("tx__")]
    X_micro = df_sub[micro_cols].copy()
    X_tx = df_sub[tx_cols].copy()
    return X_micro, X_tx

def discretize_equal_width(X: pd.DataFrame, n_bins: int = 3, verbose: bool = True):
    """
    Equal-width (uniform) discretization with KBinsDiscretizer.
    Returns: (Xb, est, const_cols)
    """
    X = X.copy()

    # 1) detect constant columns and report them without removing them
    nun = X.nunique(dropna=True)
    const_cols = nun[nun <= 1].index.tolist()

    # 2) fill missing values with the column median
    if X.isna().any().any():
        X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    # 3) discretize with equal-width bins
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")

    try:
        Z = est.fit_transform(X.values)
    except Exception as e:
        print("ERROR @ KBinsDiscretizer:", repr(e))
        print("n_bins:", n_bins)
        print("Cols input:", X.shape[1])
        print("Cols constant:", len(const_cols))
        if len(const_cols) > 0:
            print("Example of constant columns:", const_cols[:10])
        raise

    Xb = pd.DataFrame(Z, index=X.index, columns=X.columns).astype(int)

    if verbose:
        print("n_bins:", n_bins)
        print("Cols input:", X.shape[1])
        print("Cols constant found:", len(const_cols))
        if len(const_cols) > 0:
            print("Example of const cols:", const_cols[:10])
        print("Cols output:", Xb.shape[1])

    return Xb, est, const_cols

def report_empty_bins(Xb: pd.DataFrame, n_bins: int, top: int = 15) -> pd.DataFrame:
    used = Xb.nunique(dropna=True)          # bins usados por columna
    missing = n_bins - used                 # bins faltantes por columna
    out = pd.DataFrame({"bins_used": used, "bins_missing": missing})
    out = out.sort_values(["bins_missing", "bins_used"], ascending=[False, True])

    n_cols_with_empty = int((out["bins_missing"] > 0).sum())
    print("Total cols:", Xb.shape[1])
    print("Cols with >=1 empty bin :", n_cols_with_empty)
    print("bins_used:")
    print(out["bins_used"].value_counts().sort_index())
    print("\nTop columns with more empty bins:")
    print(out.head(top))
    return out

def build_MI_continuous(
    df_sub: pd.DataFrame,
    name: str,
    top_micro: int = 50,
    top_tx: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    backend: str = "threading",
) -> pd.DataFrame:
    # 1) base matrices
    X_micro_u, X_tx = get_X_by_group(df_sub)

    # 2) top variance within the group
    X_micro_small = top_by_variance(X_micro_u, top_micro)
    X_tx_small = top_by_variance(X_tx, top_tx)

    # 3) align rows
    idx = X_micro_small.index.intersection(X_tx_small.index)
    A = X_micro_small.loc[idx].copy()
    B = X_tx_small.loc[idx].copy()

    # 4) drop constant features
    A = A.loc[:, A.nunique(dropna=True) > 1]
    B = B.loc[:, B.nunique(dropna=True) > 1]

    print(f"\n== {name} ==")
    print("n samples:", df_sub.shape[0])
    print("A (micro):", A.shape, "| B (tx):", B.shape)

    # 5) MI (microbial features versus each transcript) -- PARALLEL
    MI = pd.DataFrame(index=A.columns, columns=B.columns, dtype=float)

    A_vals = A.to_numpy()
    B_vals = B.to_numpy()

    def mi_one_gene(j: int, A_vals_arg: np.ndarray, B_vals_arg: np.ndarray, seed_arg: int) -> np.ndarray:
        y = B_vals_arg[:, j]
        return mutual_info_regression(A_vals_arg, y, random_state=int(seed_arg))

    mi_list = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(mi_one_gene)(j, A_vals, B_vals, seed) for j in range(B_vals.shape[1])
    )

    MI.iloc[:, :] = np.column_stack(mi_list)

    print("MI shape:", MI.shape)
    print(MI.stack().describe())

    return MI

def edges_from_MI(MI: pd.DataFrame, pctl: float = 99.5) -> tuple[pd.DataFrame, float]:
    thr = np.nanpercentile(MI.values.flatten(), pctl)
    edges = (
        MI.stack()
        .reset_index()
        .rename(columns={"level_0": "micro_node", "level_1": "tx_node", 0: "MI"})
    )
    edges = edges[edges["MI"] >= thr].sort_values("MI", ascending=False).reset_index(drop=True)
    return edges, float(thr)

def build_bipartite_from_links(links_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for r in links_df.itertuples(index=False):
        G.add_node(r.micro_node, node_type="micro")
        G.add_node(r.tx_node, node_type="tx")
        G.add_edge(r.micro_node, r.tx_node, weight=float(r.MI))
    return G

def micro_short(name: str) -> str:
    parts = [p.strip() for p in str(name).split(";") if p.strip()]
    last = parts[-1] if parts else str(name)
    return last.replace("__", "").replace("micro__", "").strip()

def plot_hairball_color(
    G: nx.Graph,
    title: str,
    k: float = 0.8,
    seed: int = 42,
    label_micro: bool = True,
    label_tx: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Plot bipartite 'hairball' and optionally save to JPG/PNG (default 300 dpi).

    If running on a server/background job, keep show=False and use save_path.
    """
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(G, seed=seed, k=k)

    micro_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "micro"]
    tx_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "tx"]

    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.35)

    # colored nodes
    nx.draw_networkx_nodes(G, pos, nodelist=micro_nodes, node_size=450, node_color="dodgerblue")
    nx.draw_networkx_nodes(G, pos, nodelist=tx_nodes, node_size=250, node_color="seagreen")

    # labels
    labels = {}
    if label_micro:
        for n in micro_nodes:
            labels[n] = micro_short(n)
    if label_tx:
        for n in tx_nodes:
            labels[n] = str(n).replace("tx__", "")

    if labels:
        text = nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        for t in text.values():
            t.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved hairball: {save_path}")

    if show:
        plt.show()

    plt.close()

# Variables used in the analysis 
TOP_TX = 500
TOP_MICRO = 50
SEED = 42
PCTL = 99.5

N_JOBS = -1
PARALLEL_BACKEND = "threading"  # change to "loky" if you want processes

HAIRBALL_K = 0.8
HAIRBALL_SEED = 42

# discretization demo (kept from your snippet)
DISCRETIZE_DEMO = True
DEMO_TOP_MICRO = 30
DEMO_N_BINS = 3


CSV_PATH = "/Users/lorenapatriciamoraflores/Desktop/Desktop - Lorena’s MacBook Air (2)/postdoc/df_all_tx_micro_PROTEIN_CODING_ONLY.csv"


WRITE_OUTPUT = True
OUTPUT_DIR = "output_network_mi"
MI_CD_CSV = os.path.join(OUTPUT_DIR, "MI_CD.csv")
MI_UC_CSV = os.path.join(OUTPUT_DIR, "MI_UC.csv")
MI_NONIBD_CSV = os.path.join(OUTPUT_DIR, "MI_nonIBD.csv")
HAIRBALL_CD_JPG = os.path.join(OUTPUT_DIR, "hairball_CD.jpg")
HAIRBALL_UC_JPG = os.path.join(OUTPUT_DIR, "hairball_UC.jpg")
HAIRBALL_NONIBD_JPG = os.path.join(OUTPUT_DIR, "hairball_nonIBD.jpg")


def main() -> None:
    print("\n[START] network_mutual_info_ordered_from_snippet")
    print("Python:", sys.version.split()[0])
    print("CWD:", os.getcwd())
    print("CSV_PATH:", CSV_PATH, "| exists:", os.path.exists(CSV_PATH))
    sys.stdout.flush()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Fix: update CSV_PATH to the correct absolute path on your machine."
        )

    df_all = pd.read_csv(CSV_PATH)
    print("df_all:", df_all.shape)

    micro_cols = [c for c in df_all.columns if str(c).startswith("micro__")]
    tx_cols = [c for c in df_all.columns if str(c).startswith("tx__")]

    print("micro cols:", len(micro_cols))
    print("tx cols:", len(tx_cols))

    #Outputs
    
    # split feature blocks
    X_micro = df_all.filter(regex=r"^micro__").copy()
    X_tx = df_all.filter(regex=r"^tx__").copy()

    print("X_micro:", X_micro.shape)
    print("X_tx   :", X_tx.shape)

    X_micro = prep_X(X_micro)
    X_tx = prep_X(X_tx)

    # align by index in case rows were removed during preprocessing
    idx = X_micro.index.intersection(X_tx.index)
    X_micro = X_micro.loc[idx]
    X_tx = X_tx.loc[idx]

    print("After prep - X_micro:", X_micro.shape, "X_tx:", X_tx.shape)

    # global reduction used only for checks
    X_tx_small = top_by_variance(X_tx, TOP_TX)
    X_micro_small = top_by_variance(X_micro, TOP_MICRO)

    print("X_micro:", X_micro.shape)
    print("X_micro_small:", X_micro_small.shape)
    print("First 5 cols X_micro_small:")
    print(list(X_micro_small.columns[:5]))

    # check duplicate column names
    print("X_micro shape:", X_micro.shape)
    dup_mask = X_micro.columns.duplicated(keep=False)
    print("# of duplicated columns:", int(dup_mask.sum()))
    print("# of unique names:", X_micro.columns.nunique())

    # align global A/B matrices for reporting only; these are not used for the final MI calculation
    idx2 = X_micro_small.index.intersection(X_tx_small.index)
    A_global = X_micro_small.loc[idx2].copy()
    B_global = X_tx_small.loc[idx2].copy()
    print("X_micro_small:", X_micro_small.shape)
    print("A:", A_global.shape)
    print("A cols:", len(A_global.columns))

    print(df_all["diagnosis"].value_counts(dropna=False))

    # split by diagnosis
    df_cd = df_all[df_all["diagnosis"] == "CD"].copy()
    df_uc = df_all[df_all["diagnosis"] == "UC"].copy()
    df_nonibd = df_all[df_all["diagnosis"] == "nonIBD"].copy()

    # quick per-group inventory
    for name, df_sub in [("CD", df_cd), ("UC", df_uc), ("nonIBD", df_nonibd)]:
        Xm, Xt = get_X_by_group(df_sub)
        print("\n==", name, "==")
        print("n samples:", df_sub.shape[0])
        print("X_micro:", Xm.shape, "| duplicated cols:", int(Xm.columns.duplicated().sum()))
        print("X_tx:", Xt.shape)

    # within-group top-variance check
    for name, df_sub in [("CD", df_cd), ("UC", df_uc), ("nonIBD", df_nonibd)]:
        Xm, Xt = get_X_by_group(df_sub)
        Xm_small = top_by_variance(Xm, TOP_MICRO)
        Xt_small = top_by_variance(Xt, TOP_TX)
        Xm_small = Xm_small.loc[:, Xm_small.nunique(dropna=True) > 1]
        Xt_small = Xt_small.loc[:, Xt_small.nunique(dropna=True) > 1]
        print("\n==", name, "==")
        print("X_micro_small:", Xm_small.shape)
        print("X_tx_small:", Xt_small.shape)

    # discretization demo (CD only), kept from snippet
    if DISCRETIZE_DEMO:
        Xm_cd, Xt_cd = get_X_by_group(df_cd)
        Xm_small_cd = top_by_variance(Xm_cd, DEMO_TOP_MICRO)
        X_micro_bin, est_micro, const_micro = discretize_equal_width(Xm_small_cd, n_bins=DEMO_N_BINS, verbose=True)
        out_micro = report_empty_bins(X_micro_bin, n_bins=DEMO_N_BINS, top=15)

    # MI (computed ONCE per group)
    MI_CD = build_MI_continuous(df_cd, "CD", top_micro=TOP_MICRO, top_tx=TOP_TX, seed=SEED, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)
    MI_UC = build_MI_continuous(df_uc, "UC", top_micro=TOP_MICRO, top_tx=TOP_TX, seed=SEED, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)
    MI_nonIBD = build_MI_continuous(df_nonibd, "nonIBD", top_micro=TOP_MICRO, top_tx=TOP_TX, seed=SEED, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)

    print("MI equals:", MI_CD.equals(MI_UC), MI_CD.equals(MI_nonIBD), MI_UC.equals(MI_nonIBD))

    # sanity checks
    print("df_cd equals df_uc:", df_cd.equals(df_uc))
    print("df_cd equals df_nonibd:", df_cd.equals(df_nonibd))
    print("hash cd:", pd.util.hash_pandas_object(df_cd, index=True).sum())
    print("hash uc:", pd.util.hash_pandas_object(df_uc, index=True).sum())
    print("hash non:", pd.util.hash_pandas_object(df_nonibd, index=True).sum())

    # edges
    edges_CD, thr_CD = edges_from_MI(MI_CD, pctl=PCTL)
    edges_UC, thr_UC = edges_from_MI(MI_UC, pctl=PCTL)
    edges_non, thr_non = edges_from_MI(MI_nonIBD, pctl=PCTL)

    print("CD:     thr", thr_CD, "edges", edges_CD.shape[0])
    print("UC:     thr", thr_UC, "edges", edges_UC.shape[0])
    print("nonIBD: thr", thr_non, "edges", edges_non.shape[0])

    print("\nTop CD:")
    print(edges_CD.head(10))
    print("\nTop UC:")
    print(edges_UC.head(10))
    print("\nTop nonIBD:")
    print(edges_non.head(10))

    # graphs
G_CD = build_bipartite_from_links(edges_CD)
G_UC = build_bipartite_from_links(edges_UC)
G_non = build_bipartite_from_links(edges_non)

#Writing Output 

if WRITE_OUTPUT:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MI_CD.to_csv(MI_CD_CSV)
    MI_UC.to_csv(MI_UC_CSV)
    MI_nonIBD.to_csv(MI_NONIBD_CSV)
    print(f"Saved MI CSVs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
