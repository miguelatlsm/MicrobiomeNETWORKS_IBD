# This script performs Gene Ontology enrichment using gene sets derived from
# the network analysis.
# It is used to summarize the main biological functions associated with the
# condition-specific host-microbe network structure.

import networkx as nx
import community as community_louvain
import gseapy as gp
from collections import defaultdict
from networkx.algorithms import community
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# GO libraries used in this workflow:
# 'GO_Molecular_Function_2023', 'GO_Molecular_Function_2025',
# 'GO_Biological_Process_2023', and 'GO_Biological_Process_2025'.

GRAPHML_DIR = Path("MI_lasbuenas")
OUT_DIR = Path("enrichment_per_micro_from_graphml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

graphml_files = {
    "CD": GRAPHML_DIR / "B_cd_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
    "UC": GRAPHML_DIR / "B_uc_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
    "nonIBD": GRAPHML_DIR / "B_nibd_full_KEEPALLNODES_q9995_WITHMETRICS.graphml",
}

graphml_files

def load_graph(graphml_path):
    G = nx.read_graphml(graphml_path)
    print(f"Loaded: {graphml_path.name}")
    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    return G

# Remove the tx__ and micro__ prefixes before reporting results.

def strip_tx_prefix(x):
    x = str(x)
    if x.startswith("tx__"):
        return x.replace("tx__", "", 1)
    return x

def strip_micro_prefix(x):
    x = str(x)
    if x.startswith("micro__"):
        return x.replace("micro__", "", 1)
    return x

# Get node sets by class.

def get_node_sets(G):
    micro_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "micro"]
    tx_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "tx"]
    return micro_nodes, tx_nodes

# Build the background gene universe.
def get_tx_background_from_graph(G):
    _, tx_nodes = get_node_sets(G)
    tx_background = sorted({strip_tx_prefix(n) for n in tx_nodes})
    return tx_background

# Retrieve transcript neighbors for each microbial node.
def get_tx_neighbors_of_micro(G, micro_node):
    neighbors = list(G.neighbors(micro_node))
    tx_neighbors = [n for n in neighbors if G.nodes[n].get("node_type") == "tx"]
    tx_neighbors_clean = sorted({strip_tx_prefix(n) for n in tx_neighbors})
    return tx_neighbors_clean

# Run a quick check on one graph.
G_cd = load_graph(graphml_files["CD"])

micro_nodes_cd, tx_nodes_cd = get_node_sets(G_cd)
tx_background_cd = get_tx_background_from_graph(G_cd)

print("micro nodes:", len(micro_nodes_cd))
print("tx nodes:", len(tx_nodes_cd))
print("tx background:", len(tx_background_cd))
print("first 10 tx background:", tx_background_cd[:10])

# Inspect the neighborhood of one microbial node.

example_micro = micro_nodes_cd[0]
example_tx = get_tx_neighbors_of_micro(G_cd, example_micro)

print("example micro:", example_micro)
print("clean micro name:", strip_micro_prefix(example_micro))
print("n tx neighbors:", len(example_tx))
print(example_tx[:20])

# temporary compatibility patch for older gseapy on pandas 2
if not hasattr(pd.DataFrame, "append") and hasattr(pd.DataFrame, "_append"):
    pd.DataFrame.append = pd.DataFrame._append

def enrich_one_microbe(
    G,
    micro_node,
    tx_background,
    gene_sets=None,
    min_tx_neighbors=4
):
    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2025",
            "GO_Molecular_Function_2025",
        ]

    tx_genes = get_tx_neighbors_of_micro(G, micro_node)

    if len(tx_genes) < min_tx_neighbors:
        return pd.DataFrame()

    all_res = []

    for gs in gene_sets:
        try:
            enr = gp.enrichr(
                gene_list=tx_genes,
                gene_sets=gs,
                cutoff=0.05,
                outdir=None
            )

            res = enr.results.copy()
            if res.empty:
                continue

            res["micro_node"] = micro_node
            res["micro_clean"] = strip_micro_prefix(micro_node)
            res["n_tx_neighbors"] = len(tx_genes)
            res["tx_neighbors"] = ";".join(tx_genes)
            res["gene_set_library"] = gs

            all_res.append(res)

        except Exception as e:
            print(f"Error in {micro_node} | {gs}: {e}")

    if len(all_res) == 0:
        return pd.DataFrame()

    out = pd.concat(all_res, ignore_index=True)
    return out

def enrich_all_microbes_in_graph(
    G,
    network_name,
    min_tx_neighbors=4,
    gene_sets=None
):
    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2025",
            "GO_Molecular_Function_2025",
        ]

    micro_nodes, _ = get_node_sets(G)
    tx_background = get_tx_background_from_graph(G)

    print(f"\nNetwork: {network_name}")
    print("micro nodes:", len(micro_nodes))
    print("tx background:", len(tx_background))
    print("min tx neighbors:", min_tx_neighbors)

    results_all = []
    summary_rows = []

    for i, micro_node in enumerate(micro_nodes, start=1):
        tx_genes = get_tx_neighbors_of_micro(G, micro_node)

        summary_rows.append({
            "network": network_name,
            "micro_node": micro_node,
            "micro_clean": strip_micro_prefix(micro_node),
            "n_tx_neighbors": len(tx_genes)
        })

        if len(tx_genes) < min_tx_neighbors:
            continue

        if i % 100 == 0:
            print(f"Processing {i}/{len(micro_nodes)}...")

        res_micro = enrich_one_microbe(
            G=G,
            micro_node=micro_node,
            tx_background=tx_background,
            gene_sets=gene_sets,
            min_tx_neighbors=min_tx_neighbors
        )

        if not res_micro.empty:
            res_micro["network"] = network_name
            results_all.append(res_micro)

    summary_df = pd.DataFrame(summary_rows)

    if len(results_all) > 0:
        results_df = pd.concat(results_all, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    return summary_df, results_df

all_summaries = []
all_enrichments = []

for network_name, graph_path in graphml_files.items():
    print("\n" + "=" * 70)
    print(f"STARTING NETWORK: {network_name}")
    print("=" * 70)

    G = load_graph(graph_path)

    summary_df, enrichment_df = enrich_all_microbes_in_graph(
        G=G,
        network_name=network_name,
        min_tx_neighbors=4,
        gene_sets=[
            "GO_Biological_Process_2025",
            "GO_Molecular_Function_2025",

        ],

    )

    summary_path = OUT_DIR / f"microbe_neighbor_summary_{network_name}_min4_from_graphml_v2.csv"
    enrichment_path = OUT_DIR / f"resultados_enrichment_per_microbe_{network_name}_min4_from_graphml_v2.csv"

    summary_df.to_csv(summary_path, index=False)
    enrichment_df.to_csv(enrichment_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved enrichment: {enrichment_path}")

    all_summaries.append(summary_df)

    if not enrichment_df.empty:
        all_enrichments.append(enrichment_df)

    print(f"FINISHED NETWORK: {network_name}")

# optional combined outputs
combined_summary = pd.concat(all_summaries, ignore_index=True)

if len(all_enrichments) > 0:
    combined_enrichment = pd.concat(all_enrichments, ignore_index=True)
else:
    combined_enrichment = pd.DataFrame()

combined_summary.to_csv(
    OUT_DIR / "microbe_neighbor_summary_ALLNETWORKS_min4_from_graphml_v2.csv",
    index=False
)

combined_enrichment.to_csv(
    OUT_DIR / "resultados_enrichment_per_microbe_ALLNETWORKS_min4_from_graphml_v2.csv",
    index=False
)

print("\nALL DONE")
print("combined summary shape:", combined_summary.shape)
print("combined enrichment shape:", combined_enrichment.shape)


# Run enrichment on the CD network.

summary_cd, enrichment_cd = enrich_all_microbes_in_graph(
    G=G_cd,
    network_name="CD",
    min_tx_neighbors=4
)

print("summary shape:", summary_cd.shape)
print("enrichment shape:", enrichment_cd.shape)
enrichment_cd.head()

summary_cd.to_csv(
    OUT_DIR / "microbe_neighbor_summary_CD_min4_from_graphml_v1.csv",
    index=False
)

enrichment_cd.to_csv(
    OUT_DIR / "resultados_enrichment_per_microbe_CD_min4_from_graphml_v1.csv",
    index=False
)

print("Saved CD outputs")

# Run enrichment on the UC network.

summary_uc, enrichment_uc = enrich_all_microbes_in_graph(
    G=G_uc,
    network_name="UC",
    min_tx_neighbors=4
)

print("summary shape:", summary_uc.shape)
print("enrichment shape:", enrichment_uc.shape)
enrichment_uc.head()

summary_nonIBD, enrichment_nonIBD = enrich_all_microbes_in_graph(
    G=G_nonIBD,
    network_name="nonIBD",
    min_tx_neighbors=4
)

print("summary shape:", summary_nonIBD.shape)
print("enrichment shape:", enrichment_nonIBD.shape)
enrichment_nonIBD.head()

summary_uc.to_csv(
    OUT_DIR / "microbe_neighbor_summary_UC_min4_from_graphml_v1.csv",
    index=False
)

enrichment_uc.to_csv(
    OUT_DIR / "resultados_enrichment_per_microbe_UC_min4_from_graphml_v1.csv",
    index=False
)

print("Saved UC outputs")

summary_nonIBD.to_csv(
    OUT_DIR / "microbe_neighbor_summary_nonIBD_min4_from_graphml_v1.csv",
    index=False
)

enrichment_nonIBD.to_csv(
    OUT_DIR / "resultados_enrichment_per_microbe_nonIBD_min4_from_graphml_v1.csv",
    index=False
)

print("Saved nonIBD outputs")
