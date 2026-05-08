#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

files = {
    "CD": "MI_lasbuenas/sensitivity_thresholds/nodes_CD_q9995.csv",
    "UC": "MI_lasbuenas/sensitivity_thresholds/nodes_UC_q9995.csv",
    "nonIBD": "MI_lasbuenas/sensitivity_thresholds/nodes_nonIBD_q9995.csv"
}

dfs = []

for condition, path in files.items():
    df = pd.read_csv(path)
    df["network"] = condition
    dfs.append(df)

nodes = pd.concat(dfs, ignore_index=True)


# In[5]:


metrics = ["degree", "redundancy_latapy"]

node_types = ["micro", "tx"]

comparisons = [

    ("CD", "UC"),

    ("CD", "nonIBD"),

    ("UC", "nonIBD")

]

rows = []

for metric in metrics:

    for node_type in node_types:

        for a, b in comparisons:

            x = nodes.loc[

                (nodes["network"] == a) & (nodes["node_type"] == node_type),

                metric

            ].dropna()

            y = nodes.loc[

                (nodes["network"] == b) & (nodes["node_type"] == node_type),

                metric

            ].dropna()

            stat, p = ks_2samp(x, y)

            rows.append({

                "metric": metric,

                "node_type": node_type,

                "comparison": f"{a} vs {b}",

                "n_1": len(x),

                "n_2": len(y),

                "ks_statistic": stat,

                "p_value": p

            })

ks_results = pd.DataFrame(rows)

ks_results["p_adj_BH"] = multipletests(

    ks_results["p_value"],

    method="fdr_bh"

)[1]

ks_results.to_csv("KS_degree_redundancy_distributions.csv", index=False)

ks_results


# In[ ]:




