# This script compares node-level and network-level metrics across the
# condition-specific graphs generated.
# It is used to quantify changes in node relevance and network structure.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from networkx.algorithms import bipartite
import statsmodels.api as sm
import networkx as nx
import math
from collections import Counter
from pathlib import Path
import statistics

GRAPHML_DIR = Path("MI_lasbuenas/")
OUT_DIR = Path("graph_metrics_mar9_KEEPALLNODES_q9995_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPHML_DIR.resolve(), OUT_DIR.resolve()
GRAPHML_DIR.exists(), OUT_DIR.exists()

graph_files = sorted(GRAPHML_DIR.glob("*.graphml"))
len(graph_files), [f.name for f in graph_files[:20]]

wanted = {
    "B_cd_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
    "B_uc_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
    "B_nibd_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
}

graph_files_3 = [p for p in graph_files if p.name in wanted]
[p.name for p in graph_files_3]

GRAPHS = []
for fp in graph_files_3:
    G = nx.read_graphml(fp)
    GRAPHS.append((fp.stem, G))

[(name, G.number_of_nodes(), G.number_of_edges()) for name, G in GRAPHS]

for gname, G in GRAPHS:
    print("\n===", gname, "===")
    print("Graph attributes:", dict(G.graph))

for gname, G in GRAPHS:
    node_keys = set()
    for _, attrs in G.nodes(data=True):
        node_keys.update(attrs.keys())
    print("\n=== Node attribute keys:", gname, "===")
    print(sorted(node_keys))

# Assign node type from the node name prefix.

def add_prefix_type_attribute(G, attr_name="node_type"):
    for n in G.nodes():
        s = str(n).lower()  # por si viene como int u otro tipo
        if s.startswith("tx"):
            G.nodes[n][attr_name] = "tx"
        elif s.startswith("micro"):
            G.nodes[n][attr_name] = "micro"
        else:
            G.nodes[n][attr_name] = "other"

for gname, G in GRAPHS:
    add_prefix_type_attribute(G, attr_name="node_type")

for gname, G in GRAPHS:
    counts = {"tx": 0, "micro": 0, "other": 0}
    for _, attrs in G.nodes(data=True):
        counts[attrs.get("node_type", "other")] += 1
    print(gname, counts)

def nodes_df(G, redundancy_attr="redundancy_latapy"):
    rows = []
    for n, attrs in G.nodes(data=True):
        # Redundancy metric.
        x = attrs.get(redundancy_attr, np.nan)
        try:
            x = float(x)
        except Exception:
            x = np.nan

        # Degree in the current graph.
        y = G.degree(n)

        rows.append({"node": n, "redundancy_latapy": x, "degree_centrality": y})
    return pd.DataFrame(rows)

import matplotlib.pyplot as plt
for gname, G in GRAPHS:
    node_keys = set()
    for _, attrs in G.nodes(data=True):
        node_keys.update(attrs.keys())
    print("\n", gname)
    print("has redundancy:", "redundancy_latapy" in node_keys)
    print("has degree_centrality:", "degree_centrality" in node_keys)

def nodes_df_xy(G, x_attr="redundancy_latapy", y_attr="degree_centrality"):
    rows = []
    for n, attrs in G.nodes(data=True):
        x = to_float_or_nan(attrs.get(x_attr))
        y = to_float_or_nan(attrs.get(y_attr))
        rows.append({"node": n, "x": x, "y": y})
    df = pd.DataFrame(rows)
    return df

def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def plot_quadrants_real_scale(
    G, gname,
    x_attr="redundancy_latapy",
    y_attr="degree",
    zoom_percentile=99,   # 100 = min/max; 98-99 recomendado
    pad=0.08,
    jitter_y=0.12,        # 0.0 si no quieres jitter
    seed=7
):
    rng = np.random.default_rng(seed)

    rows = []
    for n, attrs in G.nodes(data=True):
        x = to_float(attrs.get(x_attr))
        y = to_float(attrs.get(y_attr))
        rows.append((n, x, y))

    df = pd.DataFrame(rows, columns=["node", "x", "y"]).dropna(subset=["x", "y"])
    print(f"{gname}: nodes={G.number_of_nodes()}  valid_points={len(df)}")

    plt.figure(figsize=(8, 7))

    if df.empty:
        plt.text(0.5, 0.5, "No valid points (check x/y attrs on nodes)",
                 transform=plt.gca().transAxes, ha="center", va="center")
        plt.title(gname)
        plt.show()
        return

    mx = float(df["x"].mean())
    my = float(df["y"].mean())

    x_plot = df["x"].to_numpy()
    y_plot = df["y"].to_numpy()

    if jitter_y and jitter_y > 0:
        y_plot = y_plot + rng.normal(0, jitter_y, size=len(y_plot))

    # Robust percentile-based limits to reduce the effect of outliers
    p = zoom_percentile
    if p >= 100:
        x_lo, x_hi = float(np.min(x_plot)), float(np.max(x_plot))
        y_lo, y_hi = float(np.min(y_plot)), float(np.max(y_plot))
    else:
        lo = (100 - p) / 2
        hi = 100 - lo
        x_lo, x_hi = np.percentile(x_plot, [lo, hi]).astype(float)
        y_lo, y_hi = np.percentile(y_plot, [lo, hi]).astype(float)

    # Center axes on the mean using symmetric limits around (mx, my)
    dx = max(abs(x_hi - mx), abs(mx - x_lo))
    dy = max(abs(y_hi - my), abs(my - y_lo))

    plt.scatter(x_plot, y_plot, alpha=0.35, s=18)

    # Quadrant lines placed at the mean
    plt.axvline(mx, linestyle="--")
    plt.axhline(my, linestyle="--")

    # Symmetric limits around the mean in the original scale
    plt.xlim(mx - dx, mx + dx)
    plt.ylim(my - dy, my + dy)

    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.title(f"{gname} — quadrants centered at means (x̄={mx:.4g}, ȳ={my:.4g})")
    plt.grid(True, alpha=0.25)
    plt.show()

for gname, G in GRAPHS:
    plot_quadrants_real_scale(
        G, gname,
        x_attr="redundancy_latapy",
        y_attr="degree",
        zoom_percentile=99,
        pad=0.08,
        jitter_y=0.12
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def plot_quadrants_with_labels(
    G, gname,
    x_attr="redundancy_latapy",
    y_attr="degree",
    label_top_n=20,
    zoom_percentile=99,
    pad=0.08,
    jitter_y=0.12,
    seed=7
):
    rng = np.random.default_rng(seed)

    # Build DF from node attributes
    rows = []
    for n, attrs in G.nodes(data=True):
        x = to_float(attrs.get(x_attr))
        y = to_float(attrs.get(y_attr))
        rows.append((n, x, y))

    df = pd.DataFrame(rows, columns=["node", "x", "y"]).dropna(subset=["x", "y"])
    print(f"{gname}: nodes={G.number_of_nodes()}  valid_points={len(df)}")

    plt.figure(figsize=(9, 8))

    if df.empty:
        plt.text(0.5, 0.5, "No valid points (check x/y attrs on nodes)",
                 transform=plt.gca().transAxes, ha="center", va="center")
        plt.title(gname)
        plt.show()
        return

    # Means for quadrant lines
    mx = float(df["x"].mean())
    my = float(df["y"].mean())

    x_plot = df["x"].to_numpy()
    y_plot = df["y"].to_numpy()

    # Jitter only for display (keeps selection based on true y)
    y_plot_j = y_plot.copy()
    if jitter_y and jitter_y > 0:
        y_plot_j = y_plot_j + rng.normal(0, jitter_y, size=len(y_plot_j))

    # Robust limits (percentiles)
    p = zoom_percentile
    if p >= 100:
        x_lo, x_hi = float(np.min(x_plot)), float(np.max(x_plot))
        y_lo, y_hi = float(np.min(y_plot_j)), float(np.max(y_plot_j))
    else:
        lo = (100 - p) / 2
        hi = 100 - lo
        x_lo, x_hi = np.percentile(x_plot, [lo, hi]).astype(float)
        y_lo, y_hi = np.percentile(y_plot_j, [lo, hi]).astype(float)

    # Symmetric limits around means (centered axes)
    dx = max(abs(x_hi - mx), abs(mx - x_lo))
    dy = max(abs(y_hi - my), abs(my - y_lo))
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    dx *= (1 + pad)
    dy *= (1 + pad)

    # Scatter
    plt.scatter(x_plot, y_plot_j, alpha=0.35, s=18)

    # Quadrant lines at means
    plt.axvline(mx, linestyle="--")
    plt.axhline(my, linestyle="--")

    plt.xlim(mx - dx, mx + dx)
    plt.ylim(my - dy, my + dy)

    # pick 20 nodes: highest degree, lowest redundancy 
    df_pick = df.sort_values(["y", "x"], ascending=[False, True]).head(label_top_n)

    # Label them
    for _, r in df_pick.iterrows():
        plt.text(r["x"], r["y"], str(r["node"]), fontsize=8)

    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.title(f"{gname} — label top {label_top_n}: high degree + low redundancy")
    plt.grid(True, alpha=0.25)
    plt.show()

for gname, G in GRAPHS:
    plot_quadrants_with_labels(
        G, gname,
        x_attr="redundancy_latapy",
        y_attr="degree",
        label_top_n=20,
        zoom_percentile=99,
        pad=0.08,
        jitter_y=0

    )

# Maximum degree within each graph.
for gname, G in GRAPHS:
    deg_attr = nx.get_node_attributes(G, "degree")
    # Cast to numeric in case the attribute was stored as text.
    vals = []
    for v in deg_attr.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    print(gname, "max_degree =", max(vals) if vals else None)

# Global maximum degree across graphs.
all_vals = []
for _, G in GRAPHS:
    for v in nx.get_node_attributes(G, "degree").values():
        try:
            all_vals.append(float(v))
        except Exception:
            pass

print("GLOBAL max_degree =", max(all_vals) if all_vals else None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def plot_quadrants_with_labels_fixed(
    G, gname,
    x_attr="redundancy_latapy",
    y_attr="degree",
    label_top_n=20,
    zoom_percentile=99,
    pad=0.08,
    jitter_y=0.12,
    seed=7
):
    rng = np.random.default_rng(seed)

    rows = []
    for n, attrs in G.nodes(data=True):
        x = to_float(attrs.get(x_attr))
        y = to_float(attrs.get(y_attr))
        rows.append((n, x, y))

    df = pd.DataFrame(rows, columns=["node", "x", "y"]).dropna(subset=["x", "y"])
    print(f"{gname}: nodes={G.number_of_nodes()}  valid_points={len(df)}  max_degree={df['y'].max()}")

    plt.figure(figsize=(9, 8))
    if df.empty:
        plt.text(0.5, 0.5, "No valid points", transform=plt.gca().transAxes, ha="center")
        plt.show()
        return

    mx = float(df["x"].mean())
    my = float(df["y"].mean())

    x_plot = df["x"].to_numpy()
    y_plot = df["y"].to_numpy()

    # Jitter y only for display
    y_plot_j = y_plot.copy()
    if jitter_y and jitter_y > 0:
        y_plot_j = y_plot_j + rng.normal(0, jitter_y, size=len(y_plot_j))

    # Map each node to its jittered y value so labels stay on the plotted point
    node_to_yj = dict(zip(df["node"].to_numpy(), y_plot_j))

    # robust limits on DISPLAY y
    p = zoom_percentile
    if p >= 100:
        x_lo, x_hi = float(np.min(x_plot)), float(np.max(x_plot))
        y_lo, y_hi = float(np.min(y_plot_j)), float(np.max(y_plot_j))
    else:
        lo = (100 - p) / 2
        hi = 100 - lo
        x_lo, x_hi = np.percentile(x_plot, [lo, hi]).astype(float)
        y_lo, y_hi = np.percentile(y_plot_j, [lo, hi]).astype(float)

    dx = max(abs(x_hi - mx), abs(mx - x_lo))
    dy = max(abs(y_hi - my), abs(my - y_lo))
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    dx *= (1 + pad)
    dy *= (1 + pad)

    plt.scatter(x_plot, y_plot_j, alpha=0.35, s=18)
    plt.axvline(mx, linestyle="--")
    plt.axhline(my, linestyle="--")
    plt.xlim(mx - dx, mx + dx)
    plt.ylim(my - dy, my + dy)

    # pick top degree + lowest redundancy (ties)
    df_pick = df.sort_values(["y", "x"], ascending=[False, True]).head(label_top_n)

    # label at the jittered y coordinate so it doesn't float away
    for _, r in df_pick.iterrows():
        plt.text(r["x"], node_to_yj[r["node"]], str(r["node"]), fontsize=8)

    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.title(f"{gname} — label top {label_top_n}: high degree + low redundancy")
    plt.grid(True, alpha=0.25)
    plt.show()

for gname, G in GRAPHS:
    plot_quadrants_with_labels_fixed(
        G, gname,
        x_attr="redundancy_latapy",
        y_attr="degree",
        label_top_n=20,
        zoom_percentile=98,
        jitter_y=0.12
    )

for name, G in GRAPHS:
    x = [G.nodes[n]["redundancy_latapy"] for n in G.nodes()]
    y = [G.nodes[n]["degree"] for n in G.nodes()]

    mx = statistics.median(x)
    my = statistics.median(y)

    plt.figure(figsize=(20, 16))
    plt.figure()

    plt.scatter(x, y)
    plt.axvline(mx)
    plt.axhline(my)
    plt.title(name)
    plt.xlabel("redundancy_latapy")
    plt.ylabel("degree")
    plt.show()

# Plot each node class separately to avoid overlap.
def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def build_df_by_type(G, node_type_value, x_attr="redundancy_latapy", y_attr="degree"):
    rows = []
    for n, attrs in G.nodes(data=True):
        if attrs.get("node_type") != node_type_value:
            continue

        x = to_float(attrs.get(x_attr))
        y = to_float(attrs.get(y_attr))
        rows.append((n, x, y))

    df = pd.DataFrame(rows, columns=["node", "x", "y"])
    return df

def get_global_limits(graphs, node_type_value, x_attr="redundancy_latapy", y_attr="degree", pad=0.05):
    dfs = []
    for gname, G in graphs:
        df = build_df_by_type(G, node_type_value, x_attr=x_attr, y_attr=y_attr)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    x_min, x_max = all_df["x"].min(), all_df["x"].max()
    y_min, y_max = all_df["y"].min(), all_df["y"].max()

    dx = (x_max - x_min) * pad if x_max > x_min else 1
    dy = (y_max - y_min) * pad if y_max > y_min else 1

    return (x_min - dx, x_max + dx, y_min - dy, y_max + dy)

def plot_set_by_node_type(graphs, node_type_value, x_attr="redundancy_latapy", y_attr="degree"):
    limits = get_global_limits(graphs, node_type_value, x_attr=x_attr, y_attr=y_attr)

    fig, axes = plt.subplots(1, len(graphs), figsize=(21, 6), sharex=True, sharey=True)

    if len(graphs) == 1:
        axes = [axes]

    for ax, (gname, G) in zip(axes, graphs):
        df = build_df_by_type(G, node_type_value, x_attr=x_attr, y_attr=y_attr)

        if df.empty:
            ax.text(0.5, 0.5, f"No data for {node_type_value}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gname)
            continue

        mx1 = df["x"].quantile(.90)
        my1 = df["y"].quantile(.10)

        ax.scatter(df["x"], df["y"], alpha=0.35, s=12)
        ax.axvline(mx1, linestyle="--")
        ax.axhline(my1, linestyle="--")
        ax.set_title(f"{gname}\n{node_type_value}")
        ax.set_xlabel(x_attr)
        ax.grid(True, alpha=0.25)
        ax.set_yscale("log")

        if limits is not None:
            ax.set_ylim(max(limits[2], 1), limits[3])
            ax.set_ylim(limits[2], limits[3])

    axes[0].set_ylabel(y_attr)
    fig.suptitle(f"{node_type_value.upper()} nodes only", fontsize=14)
    plt.tight_layout()
    plt.show()

# graph only micro
plot_set_by_node_type(
    GRAPHS,
    node_type_value="micro",
    x_attr="redundancy_latapy",
    y_attr="degree"

)

# graph only tx
plot_set_by_node_type(
    GRAPHS,
    node_type_value="tx",
    x_attr="redundancy_latapy",
    y_attr="degree"
)

for name, G in GRAPHS:
    degrees = [float(G.nodes[n].get("degree", 0)) for n in G.nodes()]
    red = [float(G.nodes[n].get("redundancy_latapy", 0)) for n in G.nodes()]

    print(f"\n{name}")
    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    print("isolates:", len(list(nx.isolates(G))))
    print("max degree:", max(degrees) if degrees else 0)
    print("mean degree:", np.mean(degrees) if degrees else 0)
    print("max redundancy:", max(red) if red else 0)
    print("mean redundancy:", np.mean(red) if red else 0)

for name, G in GRAPHS:
    micros = [n for n, d in G.nodes(data=True) if d.get("node_type") == "micro"]
    txs = [n for n, d in G.nodes(data=True) if d.get("node_type") == "tx"]

    iso = set(nx.isolates(G))

    print(f"\n{name}")
    print("micro nodes:", len(micros), "| micro isolates:", sum(n in iso for n in micros))
    print("tx nodes:", len(txs), "| tx isolates:", sum(n in iso for n in txs))

# Delta tables for node-level comparisons between networks.
def node_delta_table(df_ref, df_case, ref_name, case_name, metric):
    """
    Compare one metric node-by-node between two node metric tables.

    ref_name: e.g. 'nonIBD'
    case_name: e.g. 'CD'
    metric: one of
        'degree', 'degree_centrality', 'betweenness',
        'bipartite_clustering', 'redundancy_latapy'
    """

    cols = ["node", "type", metric]

    a = df_ref[cols].copy().rename(columns={
        metric: f"{metric}_{ref_name}",
        "type": "type_ref"
    })

    b = df_case[cols].copy().rename(columns={
        metric: f"{metric}_{case_name}",
        "type": "type_case"
    })

    # outer merge in case some node exists in one table and not another
    out = a.merge(b, on="node", how="outer")

    # unify type
    out["type"] = out["type_ref"].combine_first(out["type_case"])
    out = out.drop(columns=["type_ref", "type_case"])

    # fill missing metric values with 0
    out[f"{metric}_{ref_name}"] = out[f"{metric}_{ref_name}"].fillna(0.0)
    out[f"{metric}_{case_name}"] = out[f"{metric}_{case_name}"].fillna(0.0)

    # delta value
    out[f"delta_{metric}_{ref_name}_vs_{case_name}"] = (
        out[f"{metric}_{ref_name}"] - out[f"{metric}_{case_name}"]
    )

    # ranks within each network
    out[f"rank_{ref_name}"] = out[f"{metric}_{ref_name}"].rank(
        method="min", ascending=False
    )
    out[f"rank_{case_name}"] = out[f"{metric}_{case_name}"].rank(
        method="min", ascending=False
    )

    # positive delta_rank means node is worse ranked in case_name
    out[f"delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"rank_{case_name}"] - out[f"rank_{ref_name}"]
    )

    # optional absolute changes
    out[f"abs_delta_{metric}_{ref_name}_vs_{case_name}"] = (
        out[f"delta_{metric}_{ref_name}_vs_{case_name}"].abs()
    )
    out[f"abs_delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"delta_rank_{ref_name}_vs_{case_name}"].abs()
    )

    # nicer order
    out = out[[
        "node", "type",
        f"{metric}_{ref_name}", f"{metric}_{case_name}",
        f"delta_{metric}_{ref_name}_vs_{case_name}",
        f"rank_{ref_name}", f"rank_{case_name}",
        f"delta_rank_{ref_name}_vs_{case_name}",
        f"abs_delta_{metric}_{ref_name}_vs_{case_name}",
        f"abs_delta_rank_{ref_name}_vs_{case_name}",
    ]]

    return out.sort_values(
        by=[f"abs_delta_rank_{ref_name}_vs_{case_name}",
            f"abs_delta_{metric}_{ref_name}_vs_{case_name}"],
        ascending=[False, False]
    ).reset_index(drop=True)

def graph_to_metric_table(G: nx.Graph, graph_name: str) -> pd.DataFrame:
    rows = []

    for n, d in G.nodes(data=True):
        rows.append({
            "graph": graph_name,
            "node": n,
            "type": d.get("node_type"),
            "degree": float(d.get("degree", 0)),
            "degree_centrality": float(d.get("degree_centrality", 0)),
            "betweenness": float(d.get("betweenness", 0)),
            "bipartite_clustering": float(d.get("bipartite_clustering", 0)),
            "redundancy_latapy": float(d.get("redundancy_latapy", 0)),
            "component_id": float(d.get("component_id", -1)),
            "in_gcc": float(d.get("in_gcc", 0)),
        })

    return pd.DataFrame(rows)

tables = {}

for gname, G in GRAPHS:
    tables[gname] = graph_to_metric_table(G, gname)

tables.keys()

table_cd_full_metrics = tables["B_cd_full_KEEPALLNODES_q9995_WITHMETRICS"]
table_nibd_full_metrics = tables["B_nibd_full_KEEPALLNODES_q9995_WITHMETRICS"]
table_uc_full_metrics = tables["B_uc_full_KEEPALLNODES_q9995_WITHMETRICS"]

delta_degree_nonIBD_vs_CD = node_delta_table(
    table_nibd_full_metrics,
    table_cd_full_metrics,
    ref_name="nonIBD",
    case_name="CD",
    metric="degree"
)

delta_degree_nonIBD_vs_UC = node_delta_table(
    table_nibd_full_metrics,
    table_uc_full_metrics,
    ref_name="nonIBD",
    case_name="UC",
    metric="degree"
)

delta_degree_nonIBD_vs_CD.head(20)

METRICS = [
    "degree",
    "degree_centrality",
    "betweenness",
    "bipartite_clustering",
    "redundancy_latapy",
]

for metric in METRICS:
    df_cd_delta = node_delta_table(
        table_nibd_full_metrics,
        table_cd_full_metrics,
        ref_name="nonIBD",
        case_name="CD",
        metric=metric
    )

    df_uc_delta = node_delta_table(
        table_nibd_full_metrics,
        table_uc_full_metrics,
        ref_name="nonIBD",
        case_name="UC",
        metric=metric
    )

    df_cd_delta.to_csv(
        f"MI_lasbuenas/node_delta_{metric}_nonIBD_vs_CD_KEEPALLNODES_q9995_full.csv",
        index=False
    )

    df_uc_delta.to_csv(
        f"MI_lasbuenas/node_delta_{metric}_nonIBD_vs_UC_KEEPALLNODES_q9995_full.csv",
        index=False
    )

    print(f"Saved {metric}")

delta_degree_nonIBD_vs_CD[[
    "node", "type",
    "degree_nonIBD", "degree_CD",
    "delta_degree_nonIBD_vs_CD",
    "rank_nonIBD", "rank_CD",
    "delta_rank_nonIBD_vs_CD"
]].head(30)

# Restrict the comparison to nodes shared by both networks.
def graph_to_metric_table(G: nx.Graph, graph_name: str) -> pd.DataFrame:
    rows = []

    for n, d in G.nodes(data=True):
        rows.append({
            "graph": graph_name,
            "node": n,
            "type": d.get("node_type"),
            "degree": float(d.get("degree", 0)),
            "degree_centrality": float(d.get("degree_centrality", 0)),
            "betweenness": float(d.get("betweenness", 0)),
            "bipartite_clustering": float(d.get("bipartite_clustering", 0)),
            "redundancy_latapy": float(d.get("redundancy_latapy", 0)),
            "component_id": float(d.get("component_id", -1)),
            "in_gcc": float(d.get("in_gcc", 0)),
        })

    return pd.DataFrame(rows)

tables_short = {}

for gname, G in GRAPHS:
    gl = gname.lower()
    if "nibd" in gl:
        short = "nonIBD"
    elif "cd" in gl:
        short = "CD"
    elif "uc" in gl:
        short = "UC"
    else:
        short = gname

    tables_short[short] = graph_to_metric_table(G, short)

table_cd_full_metrics = tables_short["CD"]
table_nibd_full_metrics = tables_short["nonIBD"]
table_uc_full_metrics = tables_short["UC"]

 # Node-wise differences for shared nodes

def node_delta_table_inner(df_ref, df_case, ref_name, case_name, metric):
    """
    Compare one metric node-by-node between two node metric tables.
    Keeps ONLY nodes present in both networks.

    ref_name: e.g. 'nonIBD'
    case_name: e.g. 'CD'
    metric: one of
        'degree', 'degree_centrality', 'betweenness',
        'bipartite_clustering', 'redundancy_latapy'
    """

    cols = ["node", "type", metric]

    a = df_ref[cols].copy().rename(columns={
        metric: f"{metric}_{ref_name}",
        "type": f"type_{ref_name}"
    })

    b = df_case[cols].copy().rename(columns={
        metric: f"{metric}_{case_name}",
        "type": f"type_{case_name}"
    })

    # keep only nodes present in both networks
    out = a.merge(b, on="node", how="inner")

    # unify type
    out["type"] = out[f"type_{ref_name}"].combine_first(out[f"type_{case_name}"])
    out = out.drop(columns=[f"type_{ref_name}", f"type_{case_name}"])

    # numeric safety
    out[f"{metric}_{ref_name}"] = pd.to_numeric(out[f"{metric}_{ref_name}"], errors="coerce").fillna(0.0)
    out[f"{metric}_{case_name}"] = pd.to_numeric(out[f"{metric}_{case_name}"], errors="coerce").fillna(0.0)

    # delta of metric value
    out[f"delta_{metric}_{ref_name}_vs_{case_name}"] = (
        out[f"{metric}_{ref_name}"] - out[f"{metric}_{case_name}"]
    )

    # descending ranks inside this shared-node table
    out[f"rank_{ref_name}"] = out[f"{metric}_{ref_name}"].rank(method="min", ascending=False)
    out[f"rank_{case_name}"] = out[f"{metric}_{case_name}"].rank(method="min", ascending=False)

    # positive means better rank in ref than in case
    out[f"delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"rank_{case_name}"] - out[f"rank_{ref_name}"]
    )

    # absolute changes
    out[f"abs_delta_{metric}_{ref_name}_vs_{case_name}"] = (
        out[f"delta_{metric}_{ref_name}_vs_{case_name}"].abs()
    )
    out[f"abs_delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"delta_rank_{ref_name}_vs_{case_name}"].abs()
    )

    # nice column order
    out = out[[
        "node", "type",
        f"{metric}_{ref_name}", f"{metric}_{case_name}",
        f"delta_{metric}_{ref_name}_vs_{case_name}",
        f"rank_{ref_name}", f"rank_{case_name}",
        f"delta_rank_{ref_name}_vs_{case_name}",
        f"abs_delta_{metric}_{ref_name}_vs_{case_name}",
        f"abs_delta_rank_{ref_name}_vs_{case_name}",
    ]]

    return out.sort_values(
        by=[f"abs_delta_rank_{ref_name}_vs_{case_name}",
            f"abs_delta_{metric}_{ref_name}_vs_{case_name}"],
        ascending=[False, False]
    ).reset_index(drop=True)

# Run all metrics and save outputs

METRICS = [
    "degree",
    "degree_centrality",
    "betweenness",
    "bipartite_clustering",
    "redundancy_latapy",
]

for metric in METRICS:
    df_nonIBD_vs_CD = node_delta_table_inner(
        table_nibd_full_metrics,
        table_cd_full_metrics,
        ref_name="nonIBD",
        case_name="CD",
        metric=metric
    )

    df_nonIBD_vs_UC = node_delta_table_inner(
        table_nibd_full_metrics,
        table_uc_full_metrics,
        ref_name="nonIBD",
        case_name="UC",
        metric=metric
    )

    # full tables
    df_nonIBD_vs_CD.to_csv(
        f"MI_lasbuenas/node_delta_SHAREDONLY_{metric}_nonIBD_vs_CD_KEEPALLNODES_q9995_WITHMETRICS_v2.csv",
        index=False
    )

    df_nonIBD_vs_UC.to_csv(
        f"MI_lasbuenas/node_delta_SHAREDONLY_{metric}_nonIBD_vs_UC_KEEPALLNODES_q9995_WITHMETRICS_v2.csv",
        index=False
    )

    # top 50 by biggest rank shift
    top50_cd = df_nonIBD_vs_CD.sort_values(
        f"abs_delta_rank_nonIBD_vs_CD", ascending=False
    ).head(50)

    top50_uc = df_nonIBD_vs_UC.sort_values(
        f"abs_delta_rank_nonIBD_vs_UC", ascending=False
    ).head(50)

    top50_cd.to_csv(
        f"MI_lasbuenas/TOP50_node_delta_SHAREDONLY_{metric}_nonIBD_vs_CD_KEEPALLNODES_q9995_WITHMETRICS_v2.csv",
        index=False
    )

    top50_uc.to_csv(
        f"MI_lasbuenas/TOP50_node_delta_SHAREDONLY_{metric}_nonIBD_vs_UC_KEEPALLNODES_q9995_WITHMETRICS_v2.csv",
        index=False
    )

    print(f"Saved shared-only delta tables for {metric}")

delta_redundancy_nonIBD_vs_CD_shared = node_delta_table_inner(
    table_nibd_full_metrics,
    table_cd_full_metrics,
    ref_name="nonIBD",
    case_name="CD",
    metric="redundancy_latapy"
)

delta_redundancy_nonIBD_vs_CD_shared.head(20)

# Inspect nodes with the strongest shifts between networks.

# Check ranking changes directly.

def rank_comparison_table(df_ref, df_case, ref_name, case_name, metric):
    """
    Compare node ranks for one metric between two networks.
    Keeps only nodes present in both tables.
    """

    cols = ["node", "type", metric]

    a = df_ref[cols].copy().rename(columns={
        metric: f"{metric}_{ref_name}",
        "type": f"type_{ref_name}"
    })

    b = df_case[cols].copy().rename(columns={
        metric: f"{metric}_{case_name}",
        "type": f"type_{case_name}"
    })

    out = a.merge(b, on="node", how="inner")

    out["type"] = out[f"type_{ref_name}"].combine_first(out[f"type_{case_name}"])
    out = out.drop(columns=[f"type_{ref_name}", f"type_{case_name}"])

    out[f"{metric}_{ref_name}"] = pd.to_numeric(out[f"{metric}_{ref_name}"], errors="coerce")
    out[f"{metric}_{case_name}"] = pd.to_numeric(out[f"{metric}_{case_name}"], errors="coerce")

    # descending rank: 1 = best / highest value
    out[f"rank_{ref_name}"] = out[f"{metric}_{ref_name}"].rank(method="min", ascending=False)
    out[f"rank_{case_name}"] = out[f"{metric}_{case_name}"].rank(method="min", ascending=False)

    # positive => better in ref_name than in case_name
    out[f"delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"rank_{case_name}"] - out[f"rank_{ref_name}"]
    )

    out[f"abs_delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"delta_rank_{ref_name}_vs_{case_name}"].abs()
    )

    out = out[[
        "node", "type",
        f"{metric}_{ref_name}", f"{metric}_{case_name}",
        f"rank_{ref_name}", f"rank_{case_name}",
        f"delta_rank_{ref_name}_vs_{case_name}",
        f"abs_delta_rank_{ref_name}_vs_{case_name}",
    ]]

    return out.sort_values(
        by=f"abs_delta_rank_{ref_name}_vs_{case_name}",
        ascending=False
    ).reset_index(drop=True)

METRICS = [
    "degree",
    "degree_centrality",
    "betweenness",
    "bipartite_clustering",
    "redundancy_latapy",
]

for metric in METRICS:
    rank_nonIBD_vs_CD = rank_comparison_table(
        table_nibd_full_metrics,
        table_cd_full_metrics,
        ref_name="nonIBD",
        case_name="CD",
        metric=metric
    )

    rank_nonIBD_vs_UC = rank_comparison_table(
        table_nibd_full_metrics,
        table_uc_full_metrics,
        ref_name="nonIBD",
        case_name="UC",
        metric=metric
    )

    rank_nonIBD_vs_CD.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_KEEPALLNODES_q9995_WITHMETRICS_v3.csv",
        index=False
    )

    rank_nonIBD_vs_UC.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_KEEPALLNODES_q9995_WITHMETRICS_v3.csv",
        index=False
    )

    # top 50 strongest rank shifts
    rank_nonIBD_vs_CD.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_KEEPALLNODES_q9995_WITHMETRICS_v3.csv",
        index=False
    )

    rank_nonIBD_vs_UC.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_KEEPALLNODES_q9995_WITHMETRICS_v3.csv",
        index=False
    )

    print(f"Saved rank-shift tables for {metric}")

rank_degree_nonIBD_vs_CD = rank_comparison_table(
    table_nibd_full_metrics,
    table_cd_full_metrics,
    ref_name="nonIBD",
    case_name="CD",
    metric="degree"
)

rank_redundancy_nonIBD_vs_CD = rank_comparison_table(
    table_nibd_full_metrics,
    table_cd_full_metrics,
    ref_name="nonIBD",
    case_name="CD",
    metric="redundancy_latapy"
)

deg_part = rank_degree_nonIBD_vs_CD[
    ["node", "type", "rank_nonIBD", "rank_CD", "delta_rank_nonIBD_vs_CD"]
].rename(columns={
    "rank_nonIBD": "degree_rank_nonIBD",
    "rank_CD": "degree_rank_CD",
    "delta_rank_nonIBD_vs_CD": "degree_delta_rank_nonIBD_vs_CD"
})

red_part = rank_redundancy_nonIBD_vs_CD[
    ["node", "rank_nonIBD", "rank_CD", "delta_rank_nonIBD_vs_CD"]
].rename(columns={
    "rank_nonIBD": "redundancy_rank_nonIBD",
    "rank_CD": "redundancy_rank_CD",
    "delta_rank_nonIBD_vs_CD": "redundancy_delta_rank_nonIBD_vs_CD"
})

interesting_rank_nodes = deg_part.merge(red_part, on="node", how="inner")

# better rank in degree in nonIBD, worse rank in redundancy in nonIBD
interesting_rank_nodes = interesting_rank_nodes[
    (interesting_rank_nodes["degree_delta_rank_nonIBD_vs_CD"] > 0) &
    (interesting_rank_nodes["redundancy_delta_rank_nonIBD_vs_CD"] < 0)
].copy()

interesting_rank_nodes["combined_rank_shift_score"] = (
    interesting_rank_nodes["degree_delta_rank_nonIBD_vs_CD"].abs() +
    interesting_rank_nodes["redundancy_delta_rank_nonIBD_vs_CD"].abs()
)

interesting_rank_nodes = interesting_rank_nodes.sort_values(
    by="combined_rank_shift_score",
    ascending=False
).reset_index(drop=True)

interesting_rank_nodes.head(30)

# Split the comparison by node class.

def rank_comparison_table(df_ref, df_case, ref_name, case_name, metric):
    """
    Compare node ranks for one metric between two networks.
    Keeps only nodes present in both tables.
    """

    cols = ["node", "type", metric]

    a = df_ref[cols].copy().rename(columns={
        metric: f"{metric}_{ref_name}",
        "type": f"type_{ref_name}"
    })

    b = df_case[cols].copy().rename(columns={
        metric: f"{metric}_{case_name}",
        "type": f"type_{case_name}"
    })

    out = a.merge(b, on="node", how="inner")

    out["type"] = out[f"type_{ref_name}"].combine_first(out[f"type_{case_name}"])
    out = out.drop(columns=[f"type_{ref_name}", f"type_{case_name}"])

    out[f"{metric}_{ref_name}"] = pd.to_numeric(out[f"{metric}_{ref_name}"], errors="coerce")
    out[f"{metric}_{case_name}"] = pd.to_numeric(out[f"{metric}_{case_name}"], errors="coerce")

    # descending rank: 1 = highest value
    out[f"rank_{ref_name}"] = out[f"{metric}_{ref_name}"].rank(method="min", ascending=False)
    out[f"rank_{case_name}"] = out[f"{metric}_{case_name}"].rank(method="min", ascending=False)

    # positive => better rank in ref_name than in case_name
    out[f"delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"rank_{case_name}"] - out[f"rank_{ref_name}"]
    )

    out[f"abs_delta_rank_{ref_name}_vs_{case_name}"] = (
        out[f"delta_rank_{ref_name}_vs_{case_name}"].abs()
    )

    out = out[[
        "node", "type",
        f"{metric}_{ref_name}", f"{metric}_{case_name}",
        f"rank_{ref_name}", f"rank_{case_name}",
        f"delta_rank_{ref_name}_vs_{case_name}",
        f"abs_delta_rank_{ref_name}_vs_{case_name}",
    ]]

    return out.sort_values(
        by=f"abs_delta_rank_{ref_name}_vs_{case_name}",
        ascending=False
    ).reset_index(drop=True)

METRICS = [
    "degree",
    "degree_centrality",
    "betweenness",
    "bipartite_clustering",
    "redundancy_latapy",
]

for metric in METRICS:
    # ---------- nonIBD vs CD ----------
    rank_nonIBD_vs_CD = rank_comparison_table(
        table_nibd_full_metrics,
        table_cd_full_metrics,
        ref_name="nonIBD",
        case_name="CD",
        metric=metric
    )

    rank_nonIBD_vs_CD_micro = rank_nonIBD_vs_CD[rank_nonIBD_vs_CD["type"] == "micro"].copy()
    rank_nonIBD_vs_CD_tx = rank_nonIBD_vs_CD[rank_nonIBD_vs_CD["type"] == "tx"].copy()

    rank_nonIBD_vs_CD_micro.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_micro_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )
    rank_nonIBD_vs_CD_tx.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_tx_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )

    rank_nonIBD_vs_CD_micro.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_micro_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )
    rank_nonIBD_vs_CD_tx.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_CD_tx_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )

    # ---------- nonIBD vs UC ----------
    
    rank_nonIBD_vs_UC = rank_comparison_table(
        table_nibd_full_metrics,
        table_uc_full_metrics,
        ref_name="nonIBD",
        case_name="UC",
        metric=metric
    )

    rank_nonIBD_vs_UC_micro = rank_nonIBD_vs_UC[rank_nonIBD_vs_UC["type"] == "micro"].copy()
    rank_nonIBD_vs_UC_tx = rank_nonIBD_vs_UC[rank_nonIBD_vs_UC["type"] == "tx"].copy()

    rank_nonIBD_vs_UC_micro.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_micro_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )
    rank_nonIBD_vs_UC_tx.to_csv(
        f"MI_lasbuenas/rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_tx_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )

    rank_nonIBD_vs_UC_micro.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_micro_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )
    rank_nonIBD_vs_UC_tx.head(50).to_csv(
        f"MI_lasbuenas/TOP50_rank_shift_SHAREDONLY_{metric}_nonIBD_vs_UC_tx_KEEPALLNODES_q9995_WITHMETRICS_v4.csv",
        index=False
    )

    print(f"Saved split rank-shift tables for {metric}")

rank_redundancy_nonIBD_vs_CD = rank_comparison_table(
    table_nibd_full_metrics,
    table_cd_full_metrics,
    ref_name="nonIBD",
    case_name="CD",
    metric="redundancy_latapy"
)

rank_redundancy_nonIBD_vs_CD_micro = rank_redundancy_nonIBD_vs_CD[
    rank_redundancy_nonIBD_vs_CD["type"] == "tx"
].copy()

rank_redundancy_nonIBD_vs_CD_micro.head(20)
