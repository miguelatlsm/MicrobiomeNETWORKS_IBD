# This script filters the integrated dataset to keep protein-coding transcripts only.
# It preserves the microbiome features and metadata while restricting the
# transcript-level matrix to coding genes used in the downstream network analysis.

import pandas as pd
from biomart import BiomartServer
from io import StringIO

input_path = "df_all_tx_micro.csv"
dataset = pd.read_csv(input_path, low_memory=False)
print (dataset.shape)

server = BiomartServer("http://www.ensembl.org/biomart")

# Human dataset.
ds = server.datasets["hsapiens_gene_ensembl"]

# ds.show_filters()
# ds.show_attributes()

all_attrs = list(ds.attributes.keys())
# print(all_attrs)

# Query transcript_biotype and keep protein_coding entries.
# Run the BioMart query.

query = {
    "filters": {
        "transcript_biotype": "protein_coding"
    },
    "attributes": [
        "ensembl_gene_id",
        "external_gene_name",
        "ensembl_transcript_id",
        "transcript_biotype",
        "hgnc_symbol"
    ]
}

response = ds.search(query, header=1)

tsv = "\n".join(line.decode("utf-8") for line in response.iter_lines() if line)
df = pd.read_csv(StringIO(tsv), sep="\t")
df.head()

df.shape

query2 = {
    "filters": {
        "hgnc_symbol": "LRFN3"
    },
    "attributes": [
        "hgnc_symbol",
        "ensembl_gene_id",
        "external_gene_name",
        "ensembl_transcript_id",
        "transcript_biotype"

    ]
}

response = ds.search(query2, header=1)

tsv2 = "\n".join(line.decode("utf-8") for line in response.iter_lines() if line)
df2 = pd.read_csv(StringIO(tsv2), sep="\t")
df2.head()

df2.shape

# Take all column names from the combined table,
# keep only those starting with tx__, remove the prefix,
# and match them against the BioMart query output.

# Keep the column if the transcript ID was found; otherwise remove it.

cols = pd.Index(dataset.columns)

tx_cols = cols[cols.str.startswith("tx__")]
symbols = (
    tx_cols.to_series(index=None)
          .str.replace(r"^tx__", "", regex=True)
          .str.strip()
          .tolist()
)

print("Columns:", len(cols))
print("Columns tx__:", len(tx_cols))

mart = pd.read_csv(StringIO(tsv), sep="\t")
mart.head()

mart.shape

coding_symbols = (
    mart["HGNC symbol"]
    .dropna()
    .astype(str)
    .str.strip()
)

coding_set = set(coding_symbols)

tx_cols = [c for c in dataset.columns if isinstance(c, str) and c.startswith("tx__")]
non_tx_cols = [c for c in dataset.columns if c not in tx_cols]

tx_cols_keep = []
tx_cols_drop = []

for c in tx_cols:
    symbol = c.replace("tx__", "", 1).strip()   # quita solo el prefijo una vez
    if symbol in coding_set:
        tx_cols_keep.append(c)
    else:
        tx_cols_drop.append(c)

dataset_code = dataset[non_tx_cols + tx_cols_keep]

print("Columnas totales:", dataset.shape[1])
print("tx__ totales:", len(tx_cols))
print("tx__ que se quedan (protein_coding en mart):", len(tx_cols_keep))
print("tx__ que se van (no encontradas / raras / no codificantes):", len(tx_cols_drop))
print("Columnas finales:", dataset_code.shape[1])

print("\nEjemplos de tx__ eliminadas (primeras 30):")
print(tx_cols_drop[:30])

dataset_code.to_csv("df_all_tx_micro_PROTEIN_CODING_ONLY.csv", index=False)
