# This script converts the mutual information results into bipartite networks
# and computes the main network outputs used in downstream analyses.
# It also exports graph objects, node information, and different metrics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from networkx.algorithms import bipartite
import statsmodels.api as sm
import networkx as nx
import math
from collections import Counter

df_cd = pd.read_csv("MI_lasbuenas/MI_CD.csv", index_col=0)
df_uc = pd.read_csv("MI_lasbuenas/MI_UC.csv", index_col=0)
df_nibd = pd.read_csv("MI_lasbuenas/MI_nonIBD.csv", index_col=0)

print("CD shape:", df_cd.shape)
print("UC shape:", df_uc.shape)
print("nonIBD shape:", df_nibd.shape)

# Build the bipartite incidence matrix.

df_cd.head()

def plot_mi_distribution(df, name, bins=200, sample_size=2_000_000, seed=0):
    A = df.values.astype(float).ravel()  # volver lista la tabla
    A = A[np.isfinite(A)]                # remove NaN/inf if any

    # Print key quantiles (helps pick thresholds later)
    qs = np.quantile(A, [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0])
    print(f"\n{name} distribution (n={A.size} sampled values)")
    print("min, median, 75%, 90%, 95%, 99%, 99.5%,  99.9%, max:")
    print([float(x) for x in qs])

    # Histogram
    plt.figure()
    plt.hist(A, bins=bins)
    plt.title(f"{name}: MI distribution")
    plt.xlabel("MI")
    plt.ylabel("Count")
    plt.show()

plot_mi_distribution(df_cd, "CD")
plot_mi_distribution(df_uc, "UC")
plot_mi_distribution(df_nibd, "nonIBD")

# Filter edges by the selected threshold.

def edges_at_percentile(df, name, q=0.9995):
    A = df.values.astype(float)
    thr = float(np.quantile(A, q))

    # Filter positions where MI >= threshold
    rows, cols = np.where(A >= thr)

    edges = pd.DataFrame({
        "micro": df.index[rows],
        "tx": df.columns[cols],
        "weight": A[rows, cols]
    })

    edges = edges.sort_values("weight", ascending=False).reset_index(drop=True)

    print(f"{name} | q={q} | threshold={thr:.6f} | edges={len(edges)}")
    return thr, edges

thr_cd, edges_cd = edges_at_percentile(df_cd, "CD", q=0.9995)
thr_uc, edges_uc = edges_at_percentile(df_uc, "UC", q=0.9995)
thr_nibd, edges_nibd = edges_at_percentile(df_nibd, "nonIBD", q=0.9995)

# Optional: save for later steps
edges_cd.to_csv("breadbox_edges_CD_q9995.csv", index=False)
edges_uc.to_csv("breadbox_edges_UC_q9995.csv", index=False)
edges_nibd.to_csv("breadbox_edges_nonIBD_q9995.csv", index=False)

edges_cd.head()

# Filter the matrix and convert it to binary.
def binarize_at_threshold(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    bin_arr = (df.to_numpy(dtype=float) >= thr).astype(int)
    return pd.DataFrame(bin_arr, index=df.index, columns=df.columns)

# Use the MI matrices + their thresholds
bin_cd = binarize_at_threshold(df_cd, thr_cd)
bin_uc = binarize_at_threshold(df_uc, thr_uc)
bin_nibd = binarize_at_threshold(df_nibd, thr_nibd)

print("CD:    edges =", int(bin_cd.to_numpy().sum()), "shape =", bin_cd.shape)
print("UC:    edges =", int(bin_uc.to_numpy().sum()), "shape =", bin_uc.shape)
print("nonIBD: edges =", int(bin_nibd.to_numpy().sum()), "shape =", bin_nibd.shape)

# Optional.
bin_cd.to_csv("MI_lasbuenas/bin_MI_CD.csv")
bin_uc.to_csv("MI_lasbuenas/bin_MI_UC.csv")
bin_nibd.to_csv("MI_lasbuenas/bin_MI_nonIBD.csv")

# Output: binary matrix with the same shape, stored as 0 and 1.
# Save one filtered matrix per diagnosis group.

# Build the NetworkX graph objects.
def binary_matrix_to_bipartite_graph(bin_df: pd.DataFrame) -> nx.Graph:

    B = nx.Graph()

    micros = bin_df.index.astype(str).tolist()
    txs = bin_df.columns.astype(str).tolist()

    B.add_nodes_from(micros, bipartite="micro")
    B.add_nodes_from(txs, bipartite="tx")

    # Add edges only for 1s
    ones = bin_df.stack()
    ones = ones[ones == 1]

    for (micro, tx), _ in ones.items():
        B.add_edge(str(micro), str(tx))

    return B

B_cd = binary_matrix_to_bipartite_graph(bin_cd)
B_uc = binary_matrix_to_bipartite_graph(bin_uc)
B_nibd = binary_matrix_to_bipartite_graph(bin_nibd)

print("CD:", B_cd.number_of_nodes(), "nodes |", B_cd.number_of_edges(), "edges")
print("UC:", B_uc.number_of_nodes(), "nodes |", B_uc.number_of_edges(), "edges")
print("nonIBD:", B_nibd.number_of_nodes(), "nodes |", B_nibd.number_of_edges(), "edges")

# Add remaining nodes explicitly so the graph keeps the full node set.

# Report the number of nodes by class and the number of edges for each network.

def summarize_bipartite(B: nx.Graph):
    micros = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "micro"]
    txs = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "tx"]

    print("B nodes:", B.number_of_nodes())
    print("B edges:", B.number_of_edges())
    print("Unique micro:", len(micros))
    print("Unique tx:", len(txs))

print("=== CD ===")
summarize_bipartite(B_cd)

print("\n=== UC ===")
summarize_bipartite(B_uc)

print("\n=== nonIBD ===")
summarize_bipartite(B_nibd)

# Check that the graph is bipartite.

for name, B in [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]:
    is_bip = bipartite.is_bipartite(B)
    print(f"{name} | Is bipartite? {is_bip}")

# Count connected components in each network.

for name, B in [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]:
    components2 = list(nx.connected_components(B))
    sizes = sorted([len(c) for c in components2], reverse=True)

    print(f"\n{name}")
    print("  #components:", len(components2))
    print("  largest component size:", sizes[0] if sizes else 0)
    print("  top 5 component sizes:", sizes[:5])

# Extract the largest connected component.

for name, B in [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]:
    components2 = list(nx.connected_components(B))
    largest_size = max((len(c) for c in components2), default=0)
    print(f"{name} | largest component size: {largest_size}")

# Check whether the network contains a giant component.

for name, B in [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]:
    components2 = list(nx.connected_components(B))
    largest_size = max((len(c) for c in components2), default=0)

    threshold_nodes = math.floor(B.number_of_nodes() / 2) + 1

    print(f"\n{name}")
    print("Largest component size:", largest_size)
    print("Threshold (50% + 1):", threshold_nodes)

    if largest_size >= threshold_nodes:
        print("Hay componente gigante:)")
    else:
        print("No hay componente gigante")

def top_degree_centrality(B: nx.Graph, top_k: int = 10):
    degree_centrality2 = nx.degree_centrality(B)

    # Sort from highest to lowest. 
    top = sorted(degree_centrality2.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Top {top_k} nodes by degree centrality:")
    for node, score in top:
        node_type = B.nodes[node].get("bipartite")
        print(f"  {node_type:>5} | {node} | {score:.4f}")

# Repeat for the three networks.
for name, B in [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]:
    print(f"\n=== {name} ===")
    top_degree_centrality(B, top_k=10)

# Export ranked tables for transcript and microbial nodes.

# Save graph objects and summary tables.

# Export GraphML files.
nx.write_graphml(B_cd,   "MI_lasbuenas/B_cd_bipartite.graphml")
nx.write_graphml(B_uc,   "MI_lasbuenas/B_uc_bipartite.graphml")
nx.write_graphml(B_nibd, "MI_lasbuenas/B_nibd_bipartite.graphml")

# Export edge lists and node tables.
def export_lists(B: nx.Graph, prefix: str):
    edges = nx.to_pandas_edgelist(B)
    edges.to_csv(f"MI_lasbuenas/{prefix}_edges.csv", index=False)

    nodes = pd.DataFrame(
        [{"node": n, **d} for n, d in B.nodes(data=True)]
    )
    nodes.to_csv(f"MI_lasbuenas/{prefix}_nodes.csv", index=False)

export_lists(B_cd, "B_cd")
export_lists(B_uc, "B_uc")
export_lists(B_nibd, "B_nibd")

print("Saved: GraphML + edges.csv + nodes.csv for CD/UC/nonIBD")

# Install igraph separately if needed: pip install igraph

# Optional validation with igraph.
import igraph as ig

def nx_to_igraph(B_nx):
    nodes = list(B_nx.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in B_nx.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs["name"] = nodes
    return g

b_cd = nx_to_igraph(B_cd)
components = b_cd.connected_components()
num_components = len(components)
print(num_components)

def bipartite_node_sets(B: nx.Graph):
    micros = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "micro"]
    txs = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "tx"]
    return micros, txs

def summarize_bipartite_clustering(B: nx.Graph, name: str):
    micros, txs = bipartite_node_sets(B)

    # Bipartite clustering for each side (returns dict node -> coeff)
    cl_micro = bipartite.clustering(B, micros)
    cl_tx = bipartite.clustering(B, txs)

    v_micro = np.array(list(cl_micro.values()), dtype=float)
    v_tx = np.array(list(cl_tx.values()), dtype=float)

    print(f"\n{name}")
    print("  micro | n =", len(micros),
          "| avg =", float(v_micro.mean()) if v_micro.size else 0.0,
          "| median =", float(np.median(v_micro)) if v_micro.size else 0.0,
          "| max =", float(v_micro.max()) if v_micro.size else 0.0)

    print("    tx  | n =", len(txs),
          "| avg =", float(v_tx.mean()) if v_tx.size else 0.0,
          "| median =", float(np.median(v_tx)) if v_tx.size else 0.0,
          "| max =", float(v_tx.max()) if v_tx.size else 0.0)

# Run for the 3 networks
summarize_bipartite_clustering(B_cd, "CD")
summarize_bipartite_clustering(B_uc, "UC")
summarize_bipartite_clustering(B_nibd, "nonIBD")

# Compute Latapy bipartite clustering / redundancy metrics.

def degree_freq_by_type(B):
    # split nodes by bipartite attribute you set earlier ("micro" / "tx")
    micros = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "micro"]
    txs    = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "tx"]

    micro_deg = [B.degree(n) for n in micros]
    tx_deg    = [B.degree(n) for n in txs]

    micro_counts = Counter(micro_deg)
    tx_counts    = Counter(tx_deg)

    # sort by degree for plotting
    micro_xy = sorted(micro_counts.items())  # (degree, freq)
    tx_xy    = sorted(tx_counts.items())

    micro_x = np.array([k for k, v in micro_xy], dtype=float)
    micro_y = np.array([v for k, v in micro_xy], dtype=float)

    tx_x = np.array([k for k, v in tx_xy], dtype=float)
    tx_y = np.array([v for k, v in tx_xy], dtype=float)

    return (micro_x, micro_y), (tx_x, tx_y)

def plot_degree_distribution(ax, B, title):
    (mx, my), (tx, ty) = degree_freq_by_type(B)

    # avoid zeros on log-scale (degrees should be >=1 if isolates removed)
    ax.scatter(tx, ty, s=12, label="tx")
    ax.scatter(mx, my, s=12, label="micro")

    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.legend()

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

plot_degree_distribution(axes[0], B_cd, "CD")
plot_degree_distribution(axes[1], B_uc, "UC")
plot_degree_distribution(axes[2], B_nibd, "nonIBD")

plt.tight_layout()
plt.show()

def bipartite_node_sets(B):
    micros = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "micro"]
    txs = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "tx"]
    return micros, txs

def safe_node_redundancy(B: nx.Graph) -> dict:
    """
    Compute bipartite.node_redundancy only for nodes with degree >= 2.
    For nodes with degree < 2, set redundancy = 0.0.
    """
    rc = {n: 0.0 for n in B.nodes()}

    eligible = [n for n in B.nodes() if B.degree(n) >= 2]
    if eligible:
        rc_elig = bipartite.node_redundancy(B, nodes=eligible)
        rc.update(rc_elig)

    return rc

def summarize_redundancy(B, name):
    rc_all = safe_node_redundancy(B)

    micros, txs = bipartite_node_sets(B)
    rc_micro = np.array([rc_all[n] for n in micros], dtype=float)
    rc_tx = np.array([rc_all[n] for n in txs], dtype=float)

    print(f"\n{name}")
    print("  micro | n =", len(micros),
          "| avg =", float(rc_micro.mean()),
          "| median =", float(np.median(rc_micro)),
          "| max =", float(rc_micro.max()))
    print("    tx  | n =", len(txs),
          "| avg =", float(rc_tx.mean()),
          "| median =", float(np.median(rc_tx)),
          "| max =", float(rc_tx.max()))

    return rc_all

rc_cd = summarize_redundancy(B_cd, "CD")
rc_uc = summarize_redundancy(B_uc, "UC")
rc_nibd = summarize_redundancy(B_nibd, "nonIBD")

# Plot redundancy and export the top 10 nodes.
# Export the top 10 clustering values.

def hairball(B, title, use_largest_cc=True, remove_isolates=True, seed=7):
    G = B.copy()

    if remove_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))

    if use_largest_cc and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    print(f"{title} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=None)  # k auto; tweak if needed

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6)
    nx.draw_networkx_nodes(G, pos, node_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Hairballs (recommended: largest component + no isolates)
hairball(B_cd, "Hairball - CD", use_largest_cc=True, remove_isolates=True)
hairball(B_uc, "Hairball - UC", use_largest_cc=True, remove_isolates=True)
hairball(B_nibd, "Hairball - nonIBD", use_largest_cc=True, remove_isolates=True)

#output 
# Save the full graph with node attributes in GraphML format.
# Output summary tables.

def node_table(B: nx.Graph) -> pd.DataFrame:
    rows = []
    deg = dict(B.degree())
    dc = nx.degree_centrality(B)

    for n, d in B.nodes(data=True):
        node_type = d.get("bipartite")  # "micro" or "tx"
        rows.append({
            "node": n,
            "type": node_type,
            "degree": int(deg.get(n, 0)),
            "centrality": float(dc.get(n, 0.0))
        })

    out = pd.DataFrame(rows)

    out["node_tx"] = out.apply(lambda r: r["node"] if r["type"] == "tx" else pd.NA, axis=1)
    out["node_micro"] = out.apply(lambda r: r["node"] if r["type"] == "micro" else pd.NA, axis=1)

    out = out[["node_tx", "node_micro", "centrality", "degree", "type", "node"]]
    out = out.sort_values(["degree", "centrality"], ascending=False).reset_index(drop=True)
    return out

table_cd = node_table(B_cd)
table_uc = node_table(B_uc)
table_nibd = node_table(B_nibd)

# Save
table_cd.to_csv("MI_lasbuenas/nodes_CD_centrality_degree.csv", index=False)
table_uc.to_csv("MI_lasbuenas/nodes_UC_centrality_degree.csv", index=False)
table_nibd.to_csv("MI_lasbuenas/nodes_nonIBD_centrality_degree.csv", index=False)

print(table_cd.head(10))

def safe_node_redundancy(B: nx.Graph) -> dict:
    rc = {n: 0.0 for n in B.nodes()}
    eligible = [n for n in B.nodes() if B.degree(n) >= 2]
    if eligible:
        rc.update(bipartite.node_redundancy(B, nodes=eligible))
    return rc

def node_table_full(B: nx.Graph) -> pd.DataFrame:
    deg = dict(B.degree())
    deg_cent = nx.degree_centrality(B)
    btw = nx.betweenness_centrality(B)

    # bipartite clustering by side 
    micros = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "micro"]
    txs = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "tx"]
    cl_micro = bipartite.clustering(B, micros)
    cl_tx = bipartite.clustering(B, txs)
    bip_clust = {**cl_micro, **cl_tx}

    # Latapy redundancy 
    red = safe_node_redundancy(B)

    # components + GCC flag
    comps = list(nx.connected_components(B))
    comp_id = {}
    for i, comp in enumerate(comps):
        for n in comp:
            comp_id[n] = i
    gcc_nodes = set(max(comps, key=len)) if comps else set()

    rows = []
    for n, d in B.nodes(data=True):
        t = d.get("bipartite")
        rows.append({
            "node": n,
            "type": t,
            "degree": int(deg.get(n, 0)),
            "degree_centrality": float(deg_cent.get(n, 0.0)),
            "betweenness": float(btw.get(n, 0.0)),
            "bipartite_clustering": float(bip_clust.get(n, 0.0)),
            "redundancy_latapy": float(red.get(n, 0.0)),
            "component_id": int(comp_id.get(n, -1)),
            "in_gcc": int(n in gcc_nodes),
        })

    out = pd.DataFrame(rows)

    out["node_tx"] = np.where(out["type"] == "tx", out["node"], pd.NA)
    out["node_micro"] = np.where(out["type"] == "micro", out["node"], pd.NA)

    out = out[[
        "node_tx", "node_micro",
        "type", "degree", "degree_centrality",
        "betweenness", "bipartite_clustering", "redundancy_latapy",
        "component_id", "in_gcc", "node"
    ]].sort_values(["in_gcc", "degree", "betweenness"], ascending=[False, False, False]).reset_index(drop=True)

    return out

table_cd = node_table_full(B_cd)
table_uc = node_table_full(B_uc)
table_nibd = node_table_full(B_nibd)

table_cd.to_csv("MI_lasbuenas/nodes_CD_metrics.csv", index=False)
table_uc.to_csv("MI_lasbuenas/nodes_UC_metrics.csv", index=False)
table_nibd.to_csv("MI_lasbuenas/nodes_nonIBD_metrics.csv", index=False)

def export_graph_as_pandas_and_numpy(B, name, out_dir="MI_lasbuenas", weight=None):
    # 1) asegúrate de que el grafo trae los atributos
    attach_node_metrics(B)

    # 2) fija orden de nodos (importantísimo para que matriz y tabla hagan match)
    nodelist = list(B.nodes())

    # 3) matriz de adyacencia
    A = nx.to_numpy_array(B, nodelist=nodelist, dtype=float, weight=weight, nonedge=0.0)

    # 4) tabla de nodos (incluye TODOS los atributos)
    nodes_df = pd.DataFrame([{"node": n, **attrs} for n, attrs in B.nodes(data=True)])
    nodes_df = nodes_df.set_index("node").loc[nodelist].reset_index()  # mismo orden que A

    # 5) guarda
    np.save(f"{out_dir}/{name}_adjacency.npy", A)
    nodes_df.to_csv(f"{out_dir}/{name}_nodes_with_metrics.csv", index=False)

    # opcional: edge list con attrs
    edges_df = nx.to_pandas_edgelist(B)
    edges_df.to_csv(f"{out_dir}/{name}_edges.csv", index=False)

    return nodes_df, A

nodes_cd, A_cd = export_graph_as_pandas_and_numpy(B_cd, "CD", weight=None)
nodes_uc, A_uc = export_graph_as_pandas_and_numpy(B_uc, "UC", weight=None)
nodes_nibd, A_nibd = export_graph_as_pandas_and_numpy(B_nibd, "nonIBD", weight=None)

def export_graph_as_pandas_and_numpy(G: nx.Graph, graph_label: str, weight=None, sort_nodes=True):
    """
    Returns:
      nodes_df: one row per node with its attributes
      A: numpy adjacency matrix (n x n) using the same node order as nodes_df['node']
      edges_df: edge list as a DataFrame (with edge attributes)
    """
    nodelist = list(G.nodes())
    if sort_nodes:
        nodelist = sorted(nodelist, key=lambda x: str(x))

    nodes_df = pd.DataFrame([{"node": n, **G.nodes[n]} for n in nodelist])
    nodes_df.insert(0, "graph", graph_label)

    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight, dtype=float)

    edges_df = nx.to_pandas_edgelist(G)
    edges_df.insert(0, "graph", graph_label)

    # If you pass weight=None, networkx will create an unweighted matrix,
    # but edges_df might not have a 'weight' column. Make it explicit.
    if weight is None and "weight" not in edges_df.columns:
        edges_df["weight"] = 1.0

    return nodes_df, A, edges_df

nodes_cd, A_cd, edges_cd_df = export_graph_as_pandas_and_numpy(B_cd, "CD", weight=None)
nodes_uc, A_uc, edges_uc_df = export_graph_as_pandas_and_numpy(B_uc, "UC", weight=None)
nodes_nibd, A_nibd, edges_nibd_df = export_graph_as_pandas_and_numpy(B_nibd, "nonIBD", weight=None)

nodes_cd.head()

def top10_by_metric(B, metric, top_k=10, type_attr="type"):
    rows = []

    for n, attrs in B.nodes(data=True):
        if metric not in attrs:
            continue

        v = attrs.get(metric)

        # Skip missing / NaN
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue

        rows.append({
            "node": n,
            "type": attrs.get(type_attr, attrs.get("bipartite")),  # fallback
            "value": v
        })

    # If nothing collected, return empty DF with expected columns
    if not rows:
        return pd.DataFrame(columns=["rank", "node", "type", "value"])

    df = pd.DataFrame(rows)

    # Extra safety in case something weird happened
    if "value" not in df.columns:
        return pd.DataFrame(columns=["rank", "node", "type", "value"])

    df = df.sort_values("value", ascending=False).head(top_k).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

METRICS = [
    "degree",
    "degree_centrality",
    "betweenness",
    "bipartite_clustering",
    "redundancy_latapy",
]

GRAPHS = [("CD", B_cd), ("UC", B_uc), ("nonIBD", B_nibd)]

# Save one CSV file per metric.
for metric in METRICS:
    parts = []
    for gname, B in GRAPHS:
        df_top = top10_by_metric(B, metric, top_k=10)
        if df_top.empty:
            print(f"Skipping {metric} for {gname} (metric not found on nodes)")
            continue
        df_top.insert(0, "graph", gname)
        df_top.insert(1, "metric", metric)
        parts.append(df_top)

    if parts:
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(f"MI_lasbuenas/top10_{metric}.csv", index=False)
        print(f"Saved MI_lasbuenas/top10_{metric}.csv")
    else:
        print(f"No output for metric '{metric}' (not present in any graph)")

nx.write_graphml(B_cd,   "MI_lasbuenas/B_cd.graphml")
nx.write_graphml(B_uc,   "MI_lasbuenas/B_uc.graphml")
nx.write_graphml(B_nibd, "MI_lasbuenas/B_nonIBD.graphml")

print("Saved GraphML for CD/UC/nonIBD in MI_lasbuenas/")
