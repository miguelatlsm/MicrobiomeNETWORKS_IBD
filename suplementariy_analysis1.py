#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import platform

import pandas as pd

import numpy as np

import sklearn

import networkx as nx

import matplotlib

import gseapy

print("Python:", sys.version)

print("Platform:", platform.platform())

print("pandas:", pd.__version__)

print("numpy:", np.__version__)

print("scikit-learn:", sklearn.__version__)

print("networkx:", nx.__version__)

print("matplotlib:", matplotlib.__version__)

print("gseapy:", gseapy.__version__)


# In[2]:


import seaborn as sns

print(sns.__version__)


# In[4]:


import joblib

import scipy

import statsmodels

import igraph

import biomart

print("joblib:", joblib.__version__)

print("scipy:", scipy.__version__)

print("statsmodels:", statsmodels.__version__)

print("python-igraph:", igraph.__version__)



# In[8]:


import os
import pandas as pd
import numpy as np
from difflib import get_close_matches

OUT_DIR = "table2_mean_abundance_output"
os.makedirs(OUT_DIR, exist_ok=True)

df_path = "df_all_tx_micro.csv"

df_all = pd.read_csv(df_path, low_memory=False)

print("df_all:", df_all.shape)
print("diagnosis counts:")
print(df_all["diagnosis"].value_counts(dropna=False))


# In[9]:


#taxa for table 

table2_taxa = [micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Prevotellaceae; __Prevotella.76,
micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Porphyromonadaceae; __Odoribacter.2
micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Prevotellaceae; __Prevotella_6.10
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Lachnospiraceae_NK4A136_group.41
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Clostridiales_vadinBB60_group; __g.138
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __uncultured.1
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcaceae_UCG_014.9
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.252
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.72
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.4
micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Bacteroidales_S24_7_group; __g.42
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Clostridiales_vadinBB60_group; __g.61
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.678
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.149
micro__Bacteria; __Firmicutes; __Bacilli; __Lactobacillales; __Lactobacillaceae; __Lactobacillus.54
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.230
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; ___Eubacterium_coprostanoligenes_group.31
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcus_1.12
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Anaerostipes.1
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.269
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminiclostridium_9.6
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcaceae_UCG_014.101
micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Rikenellaceae; __Alistipes.79
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.603
micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Lachnospiraceae_NK4A136_group.50


# In[10]:


micro_cols = [c for c in df_all.columns if c.startswith("micro__")]

print("Number of micro columns:", len(micro_cols))
print("Example micro columns:")
print(micro_cols[:5])

def normalize_taxon_name(x):
    x = str(x).strip()
    x = x.replace("micro__", "")
    x = " ".join(x.split())
    return x

normalized_to_original = {
    normalize_taxon_name(c): c
    for c in micro_cols
}

matched_rows = []

for taxon in table2_taxa:
    norm_taxon = normalize_taxon_name(taxon)
    
    if norm_taxon in normalized_to_original:
        matched_col = normalized_to_original[norm_taxon]
        match_type = "exact_after_removing_micro_prefix"
        suggestions = ""
    else:
        close = get_close_matches(
            norm_taxon,
            list(normalized_to_original.keys()),
            n=5,
            cutoff=0.70
        )
        matched_col = None
        match_type = "not_found"
        suggestions = " | ".join(close)
    
    matched_rows.append({
        "table2_taxon": taxon,
        "normalized_taxon": norm_taxon,
        "matched_column": matched_col,
        "match_type": match_type,
        "suggestions": suggestions
    })

match_df = pd.DataFrame(matched_rows)

match_df.to_csv(
    f"{OUT_DIR}/Table2_taxa_column_matching.csv",
    index=False
)

match_df


# In[11]:


missing = match_df[match_df["matched_column"].isna()].copy()

print("Missing:", len(missing))

if len(missing) > 0:
    display(missing[["table2_taxon", "suggestions"]])


# In[18]:


matched_columns_manual = [
    "micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Prevotellaceae; __Prevotella.76",
    "micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Porphyromonadaceae; __Odoribacter.2",
    "micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Prevotellaceae; __Prevotella_6.10",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Lachnospiraceae_NK4A136_group.41",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Clostridiales_vadinBB60_group; __g.138",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __uncultured.1",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcaceae_UCG_014.9",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.252",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.72",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.4",
    "micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Bacteroidales_S24_7_group; __g.42",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Clostridiales_vadinBB60_group; __g.61",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.678",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.149",
    "micro__Bacteria; __Firmicutes; __Bacilli; __Lactobacillales; __Lactobacillaceae; __Lactobacillus.54",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Christensenellaceae; __Christensenellaceae_R_7_group.230",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; ___Eubacterium_coprostanoligenes_group.31",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcus_1.12",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Anaerostipes.1",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.269",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminiclostridium_9.6",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __Ruminococcaceae_UCG_014.101",
    "micro__Bacteria; __Bacteroidetes; __Bacteroidia; __Bacteroidales; __Rikenellaceae; __Alistipes.79",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Ruminococcaceae; __uncultured.603",
    "micro__Bacteria; __Firmicutes; __Clostridia; __Clostridiales; __Lachnospiraceae; __Lachnospiraceae_NK4A136_group.50",
]


# In[19]:


manual_match_df = pd.DataFrame({
    "microorganism": table2_taxa,
    "matched_column": matched_columns_manual
})

manual_match_df.to_csv(
    "table2_mean_abundance_output/manual_taxa_match_table2.csv",
    index=False
)

manual_match_df


# In[20]:


missing_manual = [
    col for col in matched_columns_manual
    if col not in df_all.columns
]

print("Missing columns:", len(missing_manual))

for col in missing_manual:
    print(col)


# In[21]:


condition_order = ["nonIBD", "CD", "UC"]

rows = []

for _, r in manual_match_df.iterrows():
    taxon = r["microorganism"]
    col = r["matched_column"]

    out = {
        "microorganism": taxon,
        "matched_column": col
    }

    for condition in condition_order:
        values = pd.to_numeric(
            df_all.loc[df_all["diagnosis"] == condition, col],
            errors="coerce"
        )

        out[f"mean_relative_abundance_{condition}"] = values.mean()
        out[f"median_relative_abundance_{condition}"] = values.median()
        out[f"n_samples_{condition}"] = values.notna().sum()

    rows.append(out)

abundance_table = pd.DataFrame(rows)

abundance_table.to_csv(
    "table2_mean_abundance_output/Table2_taxa_mean_relative_abundance_by_condition_MANUAL_MATCH.csv",
    index=False
)

abundance_table


# In[22]:


pretty_abundance = abundance_table[[
    "microorganism",
    "mean_relative_abundance_nonIBD",
    "mean_relative_abundance_CD",
    "mean_relative_abundance_UC"
]].copy()

for col in pretty_abundance.columns:
    if col != "microorganism":
        pretty_abundance[col] = pretty_abundance[col].map(
            lambda x: f"{x:.3e}" if pd.notna(x) else "NA"
        )

pretty_abundance.to_csv(
    "table2_mean_abundance_output/Table2_taxa_mean_relative_abundance_PRETTY_MANUAL_MATCH.csv",
    index=False
)

pretty_abundance


# In[14]:


def clean_name(x):
    x = str(x)
    x = x.replace("micro__", "")
    x = x.replace("_", " ")
    x = x.replace(";", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

def terminal_taxon(x):
    """
    Gets last taxonomic label.
    Example:
    Firmicutes; ...; Lachnospiraceae_NK4A136_group.41
    -> lachnospiraceae nk4a136 group 41
    """
    x = str(x).replace("micro__", "").strip()
    last = x.split(";")[-1].strip()
    last = last.replace("_", " ")
    last = re.sub(r"\s+", " ", last)
    return last.lower()

# candidate micro columns: use columns that are numeric and not metadata/transcriptomics
metadata_cols = {"diagnosis", "Participant ID", "External_ID_micro", "External_ID_trans",
                 "week_num", "visit_num", "data_type"}

candidate_cols = []
for c in df_all.columns:
    if c in metadata_cols:
        continue
    if c.startswith("tx__"):
        continue
    if pd.api.types.is_numeric_dtype(df_all[c]):
        candidate_cols.append(c)

print("Candidate numeric non-tx columns:", len(candidate_cols))
print(candidate_cols[:20])


# In[16]:


import re# Build lookup
clean_to_col = {clean_name(c): c for c in candidate_cols}
terminal_to_cols = {}

for c in candidate_cols:
    t = terminal_taxon(c)
    terminal_to_cols.setdefault(t, []).append(c)

match_rows = []

for taxon in table2_taxa:
    clean_taxon = clean_name(taxon)
    terminal = terminal_taxon(taxon)
    
    matched_col = None
    match_type = None
    suggestions = ""
    
   
    if clean_taxon in clean_to_col:
        matched_col = clean_to_col[clean_taxon]
        match_type = "exact_full_clean"
    
  
    elif terminal in terminal_to_cols and len(terminal_to_cols[terminal]) == 1:
        matched_col = terminal_to_cols[terminal][0]
        match_type = "exact_terminal_unique"
    
    elif terminal in terminal_to_cols and len(terminal_to_cols[terminal]) > 1:
        matched_col = terminal_to_cols[terminal][0]
        match_type = "exact_terminal_multiple_check"
        suggestions = " | ".join(terminal_to_cols[terminal][:10])
    
   
    else:
        close = get_close_matches(
            clean_taxon,
            list(clean_to_col.keys()),
            n=5,
            cutoff=0.55
        )
        suggestions = " | ".join([clean_to_col[x] for x in close])
        match_type = "not_found"
    
    match_rows.append({
        "table2_taxon": taxon,
        "terminal_taxon": terminal,
        "matched_column": matched_col,
        "match_type": match_type,
        "suggestions": suggestions
    })

match_df = pd.DataFrame(match_rows)
match_df.to_csv(f"{OUT_DIR}/Table2_taxa_flexible_matching.csv", index=False)
match_df


# In[17]:


match_df[match_df["matched_column"].isna()][["table2_taxon", "terminal_taxon", "suggestions"]]


# In[12]:


#relative abundance by condition

condition_order = ["nonIBD", "CD", "UC"]

rows = []

for _, r in match_df.iterrows():
    taxon = r["table2_taxon"]
    matched_col = r["matched_column"]
    
    out = {
        "microorganism": taxon,
        "matched_column": matched_col,
        "match_type": r["match_type"]
    }
    
    if pd.notna(matched_col):
        for condition in condition_order:
            values = pd.to_numeric(
                df_all.loc[df_all["diagnosis"] == condition, matched_col],
                errors="coerce"
            )
            
            out[f"mean_relative_abundance_{condition}"] = values.mean()
            out[f"median_relative_abundance_{condition}"] = values.median()
            out[f"n_samples_{condition}"] = values.notna().sum()
    else:
        for condition in condition_order:
            out[f"mean_relative_abundance_{condition}"] = np.nan
            out[f"median_relative_abundance_{condition}"] = np.nan
            out[f"n_samples_{condition}"] = 0
    
    rows.append(out)

abundance_table = pd.DataFrame(rows)

abundance_table.to_csv(
    f"{OUT_DIR}/Table2_taxa_mean_relative_abundance_by_condition.csv",
    index=False
)

abundance_table


# In[13]:


#table build

pretty_table = abundance_table[[
    "microorganism",
    "mean_relative_abundance_nonIBD",
    "mean_relative_abundance_CD",
    "mean_relative_abundance_UC"
]].copy()

for col in pretty_table.columns:
    if col != "microorganism":
        pretty_table[col] = pretty_table[col].map(
            lambda x: f"{x:.3e}" if pd.notna(x) else "NA"
        )

pretty_table.to_csv(
    f"{OUT_DIR}/Table2_taxa_mean_relative_abundance_PRETTY.csv",
    index=False
)

pretty_table


# In[ ]:




