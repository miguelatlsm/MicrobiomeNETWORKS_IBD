# This script builds the integrated analysis table used in downstream steps.
# It combines microbiome features, transcript  data, and sample metadata
# into a single dataframe aligned by sample.

import pandas as pd
import numpy as np

meta_path = "hmp2_metadata_2018-08-20.csv"

meta = pd.read_csv(meta_path, low_memory=False)
trans  = pd.read_csv(
    "host_tx_counts.tsv",
    sep="\t",
    index_col=0
)
micro = pd.read_csv(
    "taxonomic_profiles.tsv",
    sep="\t",
    index_col=False
)

print("meta:", meta.shape)
print("micro:", micro.shape)
print("trans:", trans.shape)

meta.head()

trans.head()

micro.head()

print("First Columns:", micro.columns[:5].tolist())

trans_T = trans.T

trans_T1= trans.T.reset_index().rename(columns={"index": "External_ID_trans"})

trans_T1.head()

# Remove the OTU column.
micro = micro.drop(columns=["#OTU ID"])

# Set taxonomy as the index.
micro = micro.set_index("taxonomy")
micro.index.name = "External_ID_micro"

micro.shape

micro_T = micro.T
micro_T.index.name = "External_ID_micro"

micro_T.shape

micro_T.head()

# Microbiome matrix after transpose.
d_micro = micro_T.reset_index().duplicated(keep=False).sum()
print("Exact duplicates on micro_T:", int(d_micro))

# Transcript matrix after transpose.
d_trans = trans_T1.reset_index().duplicated(keep=False).sum()
print("Exact duplicates on trans_T1:", int(d_trans))

# Metadata table.
d_meta = meta.duplicated(keep=False).sum()
print("Exact duplicates on meta:", int(d_meta))

cols = ['External ID', 'Participant ID', 'week_num', 'visit_num', 'site_sub_coll', 'data_type']

dup_count = meta.duplicated(subset=cols, keep=False).sum()
print("Duplicates on External ID, participant ID, week_num, visit_num, site_sub and datatype:", int(dup_count))

meta["data_type"].value_counts(dropna=False)

site_set  = set(meta["External ID"].astype(str).str.strip().dropna())
micro_set = set(micro_T.index.astype(str).str.strip())  # asumiendo index = External_id_micro

matches = site_set & micro_set
matches_list = sorted(matches)

print("Matches:", len(matches))
print("Matches 50 first:", matches_list[:50])
print("site_sub_coll unique:", len(site_set))
print("eExternal_ID_micro unique:", len(micro_set))

dup_count = micro_T.index.duplicated(keep=False).sum()
print("Duplicates at External_id_micro (index):", int(dup_count))

meta_matches = meta[meta["External ID"].astype(str).str.strip().isin(matches)]
print(meta_matches["data_type"].value_counts(dropna=False))

meta_set  = set(meta["External ID"].astype(str).str.strip().dropna())
trans_set = set(trans_T1["External_ID_trans"].astype(str).str.strip())

matches_tx = meta_set & trans_set
matches_tx_list = sorted(matches_tx)

print("Matches:", len(matches_tx_list))
print("Matches 50 first:", matches_tx_list[:50])
print("External ID unique (meta):", len(meta_set))
print("External_ID_trans unique (trans_T1 index):", len(trans_set))

# Check data_type values in the matched metadata rows.
meta_matches_tx = meta[meta["External ID"].astype(str).str.strip().isin(matches_tx)]
print("\nCount data_type on those matches:")
print(meta_matches_tx["data_type"].value_counts(dropna=False))

# Keep only host_transcriptomics and biopsy_16S in metadata.
keep_types = ["host_transcriptomics", "biopsy_16S"]

meta_clean = meta[meta["data_type"].astype(str).str.strip().isin(keep_types)].copy()

print("meta original:", meta.shape)
print("meta_clean:", meta_clean.shape)
print(meta_clean["data_type"].value_counts(dropna=False))

meta_set  = set(meta_clean["External ID"].astype(str).str.strip().dropna())
trans_set = set(trans_T1["External_ID_trans"].astype(str).str.strip())

matches_tx = meta_set & trans_set
matches_tx_list = sorted(matches_tx)

print("Matches:", len(matches_tx_list))
print("Matches 50 first:", matches_tx_list[:50])
print("External ID unique (meta_clean):", len(meta_set))
print("External_ID_trans unique (trans_T1 index):", len(trans_set))

# Check data_type values again after filtering.
meta_matches_tx = meta_clean[meta_clean["External ID"].astype(str).str.strip().isin(matches_tx)]
print("\nCount data_type on match (meta_clean):")
print(meta_matches_tx["data_type"].astype(str).str.strip().value_counts(dropna=False))

meta_ids_clean  = set(meta_clean["External ID"].astype(str).str.strip().dropna())
trans_ids       = set(trans_T1["External_ID_trans"].astype(str).str.strip())

missing_in_meta = sorted(trans_ids - meta_ids_clean)

print("IDs in trans_T1 that are NOT in clean :", len(missing_in_meta))
print(missing_in_meta)

meta_ids_all = set(meta["External ID"].astype(str).str.strip().dropna())
missing_in_meta_all = sorted(trans_ids - meta_ids_all)

print("IDs in trans_T1 that are NOT in meta :", len(missing_in_meta_all))
print(missing_in_meta_all)

orphans = ['CSMDRVY8', 'CSMDRVY9', 'HSm9JTBX']

trans_T1_clean = trans_T1[~trans_T1["External_ID_trans"].astype(str).str.strip().isin(orphans)].copy()

print("trans_T1 original:", trans_T1.shape)
print("trans_T1 clean (no orphans):", trans_T1_clean.shape)

meta_clean.to_csv("meta_clean.csv", index=False)
print("Saved as meta_clean.csv")

cols = ["Participant ID", "visit_num", "data_type", "week_num"]

dup_count = meta_clean.duplicated(subset=cols, keep=False).sum()
print("Duplicate rows Participant ID + visit_num + data_type:", int(dup_count))

dups = meta_clean.loc[meta_clean.duplicated(subset=cols, keep=False), cols].sort_values(cols)
dups.head(50)

key = ["Participant ID","week_num","visit_num","data_type"]

grp = meta_clean.groupby(key).size().reset_index(name="n")
dups = grp[grp["n"] > 1].sort_values("n", ascending=False)

print("Duplicate groups (n>1):", dups.shape[0])
print("Top 20:")
print(dups.head(50))

key = ["Participant ID","week_num","visit_num","data_type"]

grp = meta_clean.groupby(key).size().reset_index(name="n")
print("Duplicate groups (n>1):", (grp["n"]>1).sum())
print("Total rows in duplicates:", int(grp.loc[grp["n"]>1, "n"].sum()))

key = ["Participant ID","week_num","visit_num","data_type"]

diag = (meta_clean.groupby(key)
        .agg(n_rows=("External ID","size"),
             n_external_ids=("External ID","nunique"))
        .reset_index()
        .sort_values(["n_rows","n_external_ids"], ascending=False))

print(diag.head(20))

keyran = ["Participant ID","week_num","visit_num","data_type"]

meta_dedup_first = (
    meta_clean
    .drop_duplicates(subset=keyran, keep="first")
)

print("meta_clean:", meta_clean.shape)
print("meta_dedup_first:", meta_dedup_first.shape)
print("Duplicates after:", meta_dedup_first.duplicated(subset=keyran, keep=False).sum())

# Check repeated samples across visits for the same participant and data type.

keyex = ["Participant ID", "data_type"]

grp = meta_dedup_first.groupby(keyex).size().reset_index(name="n")
dups = grp[grp["n"] > 1].sort_values("n", ascending=False)

print("Groups with more than one visit (n>1):", dups.shape[0])
print(dups.head(20))

# Keep the earliest paired visit to reduce treatment-related variation.

key_visit = ["Participant ID", "week_num", "visit_num"]
keep_types = {"host_transcriptomics", "biopsy_16S"}

# 1) Identify visits with both data types present.
v = (meta_dedup_first[meta_dedup_first["data_type"].isin(keep_types)]
     .groupby(key_visit)["data_type"].nunique()
     .reset_index(name="n_types"))

both = v[v["n_types"] == 2].copy()

# 2) For each participant, select the earliest visit based on week_num and visit_num.
first_visit = (both.sort_values(["Participant ID", "week_num", "visit_num"])
               .drop_duplicates(subset=["Participant ID"], keep="first"))

print("Participants with at least onw visit in both:", first_visit.shape[0])

# 3) Keep those visits and retain both rows: host_transcriptomics and biopsy_16S.
meta_first_pair = (meta_dedup_first
                   .merge(first_visit[key_visit], on=key_visit, how="inner")
                   .query("data_type in ['host_transcriptomics','biopsy_16S']")
                   .copy())

print("Result (Should be 2 rows per participant):", meta_first_pair.shape)
print(meta_first_pair["data_type"].value_counts(dropna=False))

# Confirm that each participant contributes one paired visit.
chk = meta_first_pair.groupby("Participant ID").size().value_counts()
print(chk)   # idealmente todo debe ser 2

dx_counts = (
    meta_first_pair
    .drop_duplicates(subset=["Participant ID"])
    ["diagnosis"]
    .value_counts(dropna=False)
)

dx_counts

def _clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _clean_index(idx: pd.Index) -> pd.Index:
    return pd.Index(idx.astype(str)).str.strip()

key_visit = ["Participant ID", "week_num", "visit_num"]

# 1) Prepare microbiome features with External_ID_micro as index.

micro_tbl = micro_T.copy()

micro_tbl.index = _clean_index(micro_tbl.index)
micro_tbl.index.name = "External_ID_micro"

# Optional sanity check.
print("micro_tbl index name:", micro_tbl.index.name)
print("micro_tbl duplicated index:", int(micro_tbl.index.duplicated().sum()))

# Add a prefix to avoid name collisions.
micro_tbl = micro_tbl.add_prefix("micro__")

# 2) Prepare transcript features with External_ID_trans as index.
#    If it is still a column, move it to the index.

trans_tbl = trans_T1_clean.copy()

if "External_ID_trans" in trans_tbl.columns:
    trans_tbl["External_ID_trans"] = _clean_series(trans_tbl["External_ID_trans"])
    trans_tbl = trans_tbl.set_index("External_ID_trans", drop=True)

trans_tbl.index = _clean_index(trans_tbl.index)
trans_tbl.index.name = "External_ID_trans"

print("trans_tbl index name:", trans_tbl.index.name)
print("trans_tbl duplicated index:", int(trans_tbl.index.duplicated().sum()))

# Add a prefix to avoid name collisions.
trans_tbl = trans_tbl.add_prefix("tx__")

# 3) Split metadata into transcript and microbiome branches.

meta2 = meta_clean.copy()
meta2["External ID"] = _clean_series(meta2["External ID"])
meta2["data_type"] = meta2["data_type"].astype(str).str.strip()

meta_tx = (meta2[meta2["data_type"] == "host_transcriptomics"]
           [key_visit + ["External ID"]]
           .rename(columns={"External ID": "External_ID_trans"})
           .drop_duplicates())

meta_micro = (meta2[meta2["data_type"] == "biopsy_16S"]
              [key_visit + ["External ID"]]
              .rename(columns={"External ID": "External_ID_micro"})
              .drop_duplicates())

print("meta_tx:", meta_tx.shape, " | meta_micro:", meta_micro.shape)

# 4) Merge both branches in parallel using metadata keys.

tx_branch = meta_tx.merge(trans_tbl, left_on="External_ID_trans", right_index=True, how="inner")
micro_branch = meta_micro.merge(micro_tbl, left_on="External_ID_micro", right_index=True, how="inner")

print("tx_branch:", tx_branch.shape)
print("micro_branch:", micro_branch.shape)

# One final row per visit containing both omics layers.

micro_cols = [c for c in micro_branch.columns if c.startswith("micro__")]
micro_only = micro_branch[key_visit + ["External_ID_micro"] + micro_cols].copy()

df_all = tx_branch.merge(micro_only, on=key_visit, how="inner")

print("df_all:", df_all.shape)
print("Participantes:", df_all["Participant ID"].nunique())

# Check for duplicates in the final visit-level table.
print("Duplicates per visit @ df_all:", int(df_all.duplicated(subset=key_visit).sum()))

seed = 42
rng = np.random.default_rng(seed)

keep_types = ["host_transcriptomics", "biopsy_16S"]
key_visit = ["Participant ID", "week_num", "visit_num"]
key_pick  = ["Participant ID", "week_num", "visit_num", "data_type"]

m = meta_clean.copy()
m.columns = m.columns.astype(str).str.strip()
m["data_type"] = m["data_type"].astype(str).str.strip()
m["External ID"] = m["External ID"].astype(str).str.strip()

# 0) Keep only the two target data types.
m = m[m["data_type"].isin(keep_types)].copy()

# 1) Find visits where both data types are available.
both = (m.groupby(key_visit)["data_type"].nunique().reset_index(name="n_types"))
both = both[both["n_types"] == 2].copy()

# 2) Select the earliest visit for each participant.
first_visit = (both.sort_values(["Participant ID", "week_num", "visit_num"])
                  .drop_duplicates(subset=["Participant ID"], keep="first"))

# 3) Keep only rows from that earliest visit.
m_first = m.merge(first_visit[key_visit], on=key_visit, how="inner").copy()

# 4) If duplicate rows remain within a visit, keep one row at random per data type.
m_first["_r"] = rng.random(len(m_first))
meta_pair = (m_first.sort_values(key_pick + ["_r"])
                    .drop_duplicates(subset=key_pick, keep="first")
                    .drop(columns="_r"))

print("meta_pair shape:", meta_pair.shape)
print(meta_pair["data_type"].value_counts())

# Sanity check: ideally two rows per participant, one per data type.
print(meta_pair.groupby("Participant ID").size().value_counts())

key_visit = ["Participant ID","week_num","visit_num"]

# --- 1) Split branches and rename external IDs ---
meta_tx = meta_pair[meta_pair["data_type"]=="host_transcriptomics"].rename(columns={"External ID":"External_ID_trans"}).copy()
meta_micro = meta_pair[meta_pair["data_type"]=="biopsy_16S"].rename(columns={"External ID":"External_ID_micro"}).copy()

# --- 2) Microbiome features (index = External_ID_micro) ---
micro_tbl = micro_T.copy()
micro_tbl.index = micro_tbl.index.astype(str).str.strip()
micro_tbl.index.name = "External_ID_micro"
micro_tbl = micro_tbl.add_prefix("micro__")

# --- 3) Transcript features (force External_ID_trans to the index) ---
trans_tbl = trans_T1.copy()
if "External_ID_trans" in trans_tbl.columns:
    trans_tbl["External_ID_trans"] = trans_tbl["External_ID_trans"].astype(str).str.strip()
    trans_tbl = trans_tbl.set_index("External_ID_trans")

trans_tbl.index = trans_tbl.index.astype(str).str.strip()
trans_tbl.index.name = "External_ID_trans"
trans_tbl = trans_tbl.add_prefix("tx__")

# --- 4) Parallel merge by external ID ---
tx_branch = meta_tx.merge(trans_tbl, left_on="External_ID_trans", right_index=True, how="inner")
micro_branch = meta_micro.merge(micro_tbl, left_on="External_ID_micro", right_index=True, how="inner")

print("tx_branch:", tx_branch.shape)
print("micro_branch:", micro_branch.shape)

# --- 5) Prepare fields added from the microbiome branch ---
# A) Microbiome features.
micro_feat_cols = [c for c in micro_branch.columns if c.startswith("micro__")]

# B) Microbiome metadata not already present in the transcript branch.
micro_meta_cols = [
    c for c in micro_branch.columns
    if (c not in key_visit)
    and (c not in micro_feat_cols)
    and (c != "External_ID_micro")
    and (c != "data_type")
    and (c not in tx_branch.columns)
]

micro_only = micro_branch[key_visit + ["External_ID_micro"] + micro_meta_cols + micro_feat_cols].copy()

# Prefix microbiome metadata so its source remains explicit.
rename_micro_meta = {c: f"micro_meta__{c}" for c in micro_meta_cols}
micro_only = micro_only.rename(columns=rename_micro_meta)

# --- 6) Final table: keep full transcript metadata and add microbiome fields ---
# The transcript branch already contains the original metadata plus tx__ features.
df_all = tx_branch.merge(micro_only, on=key_visit, how="inner", validate="1:1")

print("df_all:", df_all.shape)
print("Duplicates per visit:", df_all.duplicated(subset=key_visit, keep=False).sum())
print("Participants:", df_all["Participant ID"].nunique())

# Save the merged table to CSV.
out_path = "df_all_tx_micro.csv"
df_all.to_csv(out_path, index=False)

print("Saved @:", out_path)
