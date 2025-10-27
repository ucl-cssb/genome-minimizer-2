import os
import json
import time
import logging
import numpy as np
import pandas as pd

from ..utils.directories import (
    TEN_K_DATASET_FULL,
    SEQUENCES_FULL,
    SEQUENCE_OUT,
    PAPER_ESSENTIAL_GENES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_files():
    essential_genes = pd.read_csv(PAPER_ESSENTIAL_GENES)
    essential_set = set(essential_genes["# gene"].astype(str))

    id_lists = np.load(SEQUENCE_OUT, allow_pickle=True)

    return essential_set, id_lists

def masks_to_gene_lists(
    masks_npy_path: str,
    cols, 
    out_ids_npy: str,
    threshold: float = 0.5,
):
    logging.info(f"masks: {masks_npy_path}")
    P = len(cols)
    logging.info(f"Resolved {P} gene columns")

    uniq, first_idx = np.unique(cols, return_index=True)
    if len(uniq) != P:
        dup_count = P - len(uniq)
        logging.warning(f"{dup_count} duplicate gene names detected; keeping first occurrences")
        keep_mask = np.zeros(P, dtype=bool)
        keep_mask[np.sort(first_idx)] = True
        cols = cols[keep_mask]
        P = len(cols)

    logging.info("Loading masks .npy ...")
    masks = np.load(masks_npy_path, allow_pickle=True)
    if masks.ndim == 1:
        if isinstance(masks[0], (list, np.ndarray)):
            masks = np.array([np.asarray(row) for row in masks], dtype=object)
        else:
            masks = masks[None, :]
    N = len(masks)
    logging.info(f"Masks shape: N={N} samples")

    rows = []
    for i, row in enumerate(masks):
        r = np.asarray(row, dtype=float)
        if r.size != P:
            msg = f"Mask row {i} has length {r.size}, but dataset has {P} gene columns."
            logging.error(msg)
            raise ValueError(msg)
        rows.append((r >= threshold).astype(bool))
        if (i + 1) % 10 == 0:
            print(f"[progress] thresholded {i+1}/{N} masks")

    M = np.stack(rows, axis=0) 
    logging.info("Thresholding complete")

    id_lists = []
    for i in range(N):
        genes_present = cols[M[i]].tolist()
        id_lists.append(genes_present)
        if (i + 1) % 10 == 0:
            print(f"[progress] converted {i+1}/{N} masks to gene lists")

    if out_ids_npy:
        os.makedirs(os.path.dirname(out_ids_npy) or ".", exist_ok=True)
        np.save(out_ids_npy, np.array(id_lists, dtype=object))
        logging.info(f"Saved IDs (NPY): {out_ids_npy}")

    sizes = np.fromiter((len(L) for L in id_lists), dtype=int)

    print(f"✓ Number of samples processed = {N} | Average gene count = {sizes.mean():.1f}")

def check_essential_genes(essential_set, id_lists):
    total_essential = len(essential_set)
    n_samples = len(id_lists)
    logging.info(f"Checking & fixing essential genes (n={total_essential}) across {n_samples} samples")

    updated_samples = []
    n_fixed = 0
    n_ok = 0

    for idx, gene_list in enumerate(id_lists):
        if isinstance(gene_list, np.ndarray):
            gene_list = gene_list.tolist()

        gene_set = set(gene_list)
        # print(len(gene_set))
        missing = essential_set - gene_set

        if missing:
            logging.warning(f"Sample {idx+1} is missing {len(missing)} essential genes; adding them")
            gene_set.update(missing)
            missing_after = essential_set - gene_set
            # print(len(essential_set))
            # print(len(gene_set))
            if missing_after:
                raise RuntimeError(
                    f"Post-add verify failed for sample {idx+1}: still missing {len(missing_after)} essentials "
                    f"(e.g., {list(missing_after)[:5]} ...)"
                )
            n_fixed += 1
        else:
            n_ok += 1

        updated_samples.append(sorted(gene_set))

        if (idx + 1) % 10 == 0:
            print(f"[progress] verified {idx+1}/{n_samples} samples")

    base, ext = os.path.splitext(SEQUENCE_OUT)
    out_path = base + "_with_essentials" + ext
    np.save(out_path, np.array(updated_samples, dtype=object))
    logging.info(f"Saved updated samples with essential genes to: {out_path}")

    print(f"✓ Verified {n_samples} samples | already OK: {n_ok} | fixed: {n_fixed}")
    return out_path


def main():
    large_data = pd.read_csv(TEN_K_DATASET_FULL, index_col=0)
    data_without_lineage = large_data.drop(index=["Lineage"], errors="ignore")
    data_transpose = data_without_lineage.transpose()
    print(f"Dataset shape (samples x genes): {data_transpose.shape}")
    cols = data_transpose.columns

    masks_to_gene_lists(
        masks_npy_path=SEQUENCES_FULL,
        cols=cols,  
        out_ids_npy=SEQUENCE_OUT,
    )
    essential_set, id_lists = load_files()
    check_essential_genes(essential_set, id_lists)

if __name__ == "__main__":
    main()
