import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, alt, json, mo, np, pd, sys


@app.cell
def _(mo):
    mo.md(r"""
    # v3 latent-space diagnostics

    Minimal diagnostics for reviewer questions about whether the **v3 VAE**
    uses its latent space.

    This notebook reports:

    1. **Active latent components** using per-dimension KL and variance of
       encoder means.
    2. **Decoder sensitivity to latent input** by sampling from the prior
       and measuring variation in decoded genomes.
    3. **Posterior sampling from parental genomes** by sampling from
       `q(z | x_parent)` and writing posterior-derived gene lists for
       downstream FBA / BioGRID screening.

    Interpretation:

    - If KL is close to zero for all latent dimensions, the latent space is
      likely unused.
    - If prior-decoded genomes are nearly identical, the decoder is
      insensitive to `z`.
    - If posterior samples from parental genomes perform better downstream
      than prior samples at matched size, that supports the model capturing
      genome-context or lineage constraints.
    """)
    return


@app.cell
def _(Path, sys):
    # Assumes this file lives in repo_root/notebooks/.
    # If you move it elsewhere, edit PROJECT_ROOT manually.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    EVAL_DATA = PROJECT_ROOT / "evaluation" / "data"
    OUT_DIR = EVAL_DATA / "v3_latent_diagnostics"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"DATA_DIR     = {DATA_DIR}")
    print(f"OUT_DIR      = {OUT_DIR}")
    return DATA_DIR, OUT_DIR, PROJECT_ROOT


@app.cell
def _(mo):
    n_real = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=500,
        label="Number of real genomes to encode",
    )

    n_prior = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=500,
        label="Number of prior samples",
    )

    n_parents = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=20,
        label="Number of parental genomes",
    )

    n_posterior_per_parent = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=20,
        label="Posterior samples per parent",
    )

    decode_threshold = mo.ui.slider(
        start=0.30,
        stop=0.70,
        step=0.01,
        value=0.50,
        label="Decode threshold",
    )

    mo.vstack(
        [
            n_real,
            n_prior,
            n_parents,
            n_posterior_per_parent,
            decode_threshold,
        ]
    )
    return decode_threshold, n_parents, n_posterior_per_parent, n_prior, n_real


@app.cell
def _(DATA_DIR, n_real, np, pd):
    def load_real_binary_matrix(pa_csv_path, n_genomes, seed=42):
        """Load a random subset of real genomes.

        F4_complete_presence_absence.csv is genes x strains, with a Lineage row.
        Returns:
            X_real_np: n_genomes x n_genes binary matrix
            gene_order: gene names in model/input order
            strain_names: selected strain IDs
        """
        _header = pd.read_csv(pa_csv_path, nrows=0).columns.tolist()
        _strain_cols = _header[1:]

        _rng = np.random.default_rng(seed)
        _chosen_idx = _rng.choice(
            len(_strain_cols),
            size=min(int(n_genomes), len(_strain_cols)),
            replace=False,
        )
        _chosen_names = [_strain_cols[_i] for _i in sorted(_chosen_idx)]

        _df = pd.read_csv(
            pa_csv_path,
            index_col=0,
            usecols=[_header[0], *_chosen_names],
        )
        _df = _df.drop(index="Lineage", errors="ignore")

        _gene_order = _df.index.to_numpy()
        _X = _df.T.astype(np.float32).to_numpy()

        return _X, _gene_order, _chosen_names

    X_real_np, gene_order, strain_names = load_real_binary_matrix(
        DATA_DIR / "F4_complete_presence_absence.csv",
        n_genomes=n_real.value,
        seed=42,
    )

    print(f"Loaded real genomes: {X_real_np.shape[0]}")
    print(f"Input genes: {X_real_np.shape[1]}")
    return X_real_np, gene_order, strain_names


@app.cell
def _(X_real_np):
    import torch
    from huggingface_hub import hf_hub_download
    from genome_minimizer_2.training.model import VAE

    V3_HIDDEN_DIM = 512
    V3_LATENT_DIM = 32
    V3_REPO_ID = "UCL-CSSB/genome-minimizer-2"
    V3_BRANCH = "v3"

    _ckpt_path = hf_hub_download(V3_REPO_ID, "final.pt", revision=V3_BRANCH)

    v3_model = VAE(
        X_real_np.shape[1],
        V3_HIDDEN_DIM,
        V3_LATENT_DIM,
    )

    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
    v3_model.load_state_dict(_ckpt["model_state_dict"])
    v3_model.eval()

    print(f"Loaded v3 checkpoint: {_ckpt_path}")
    print(f"Epoch: {_ckpt.get('epoch', 'unknown')}")
    print(f"Latent dim: {V3_LATENT_DIM}")
    return V3_LATENT_DIM, torch, v3_model


@app.cell
def _(v3_model):
    def v3_get_mu_logvar(x_batch):
        """Robustly extract encoder mean and log-variance."""
        if hasattr(v3_model, "encode"):
            _out = v3_model.encode(x_batch)
            if isinstance(_out, tuple) and len(_out) >= 2:
                return _out[0], _out[1]

        _out = v3_model(x_batch)
        if isinstance(_out, tuple) and len(_out) >= 3:
            return _out[1], _out[2]

        raise RuntimeError(
            "Could not extract mu/logvar. Expected model.encode(x)->(mu, logvar) "
            "or model(x)->(recon, mu, logvar)."
        )

    return (v3_get_mu_logvar,)


@app.cell
def _(X_real_np, np, pd, torch, v3_get_mu_logvar):
    """Encode real genomes and compute active latent dimensions."""
    _X = torch.tensor(X_real_np, dtype=torch.float32)

    _mu_batches = []
    _logvar_batches = []
    _batch_size = 64

    with torch.no_grad():
        for _start in range(0, _X.shape[0], _batch_size):
            _xb = _X[_start : _start + _batch_size]
            _mu_b, _logvar_b = v3_get_mu_logvar(_xb)
            _mu_batches.append(_mu_b.cpu())
            _logvar_batches.append(_logvar_b.cpu())

    v3_mu_np = torch.cat(_mu_batches, dim=0).numpy()
    v3_logvar_np = torch.cat(_logvar_batches, dim=0).numpy()

    _var_np = np.exp(v3_logvar_np)
    _sigma_np = np.sqrt(_var_np)

    # Per-sample, per-dimension KL:
    # KL(q(z_j|x) || N(0,1)) = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
    _kl_per_sample_dim = 0.5 * (
        v3_mu_np**2 + _var_np - v3_logvar_np - 1.0
    )

    _mean_kl_dim = _kl_per_sample_dim.mean(axis=0)
    _var_mu_dim = v3_mu_np.var(axis=0)
    _mean_abs_mu_dim = np.abs(v3_mu_np).mean(axis=0)
    _mean_sigma_dim = _sigma_np.mean(axis=0)

    active_kl_threshold = 0.01
    active_var_mu_threshold = 0.01

    v3_active_latent_df = (
        pd.DataFrame(
            {
                "latent_dim": np.arange(v3_mu_np.shape[1]),
                "mean_kl_nats": _mean_kl_dim,
                "var_mu": _var_mu_dim,
                "mean_abs_mu": _mean_abs_mu_dim,
                "mean_sigma": _mean_sigma_dim,
                "active_by_kl": _mean_kl_dim > active_kl_threshold,
                "active_by_var_mu": _var_mu_dim > active_var_mu_threshold,
                "active_by_either": (
                    (_mean_kl_dim > active_kl_threshold)
                    | (_var_mu_dim > active_var_mu_threshold)
                ),
            }
        )
        .sort_values("mean_kl_nats", ascending=False)
        .reset_index(drop=True)
    )

    v3_active_summary = {
        "active_kl_threshold": active_kl_threshold,
        "active_var_mu_threshold": active_var_mu_threshold,
        "n_active_by_kl": int(v3_active_latent_df["active_by_kl"].sum()),
        "n_active_by_var_mu": int(v3_active_latent_df["active_by_var_mu"].sum()),
        "n_active_by_either": int(v3_active_latent_df["active_by_either"].sum()),
        "total_mean_kl_nats": float(_mean_kl_dim.sum()),
        "mean_kl_per_dim_nats": float(_mean_kl_dim.mean()),
    }

    print(v3_active_summary)
    v3_active_latent_df.head(10)
    return v3_active_latent_df, v3_active_summary


@app.cell
def _(alt, v3_active_latent_df):
    _plot_df = v3_active_latent_df.sort_values("latent_dim")

    alt.Chart(_plot_df).mark_bar().encode(
        x=alt.X("latent_dim:O", title="latent dimension"),
        y=alt.Y("mean_kl_nats:Q", title="mean KL contribution (nats)"),
        tooltip=[
            "latent_dim",
            "mean_kl_nats",
            "var_mu",
            "mean_abs_mu",
            "mean_sigma",
            "active_by_kl",
            "active_by_var_mu",
        ],
    ).properties(
        width=520,
        height=260,
        title="v3 active latent dimensions: per-dimension KL",
    )
    return


@app.cell
def _(alt, v3_active_latent_df):
    _plot_df = v3_active_latent_df.sort_values("latent_dim")

    alt.Chart(_plot_df).mark_bar().encode(
        x=alt.X("latent_dim:O", title="latent dimension"),
        y=alt.Y("var_mu:Q", title="variance of encoder mean"),
        tooltip=[
            "latent_dim",
            "mean_kl_nats",
            "var_mu",
            "mean_abs_mu",
            "mean_sigma",
            "active_by_kl",
            "active_by_var_mu",
        ],
    ).properties(
        width=520,
        height=260,
        title="v3 active latent dimensions: variance of encoder means",
    )
    return


@app.cell
def _(np):
    def v3_pairwise_jaccard_summary(binary_matrix, max_items=300, seed=42):
        """Estimate pairwise Jaccard distance among binary decoded genomes."""
        _rng = np.random.default_rng(seed)
        _n = binary_matrix.shape[0]

        if _n > max_items:
            _idx = _rng.choice(_n, size=max_items, replace=False)
            _B = binary_matrix[_idx].astype(bool)
        else:
            _B = binary_matrix.astype(bool)

        _distances = []
        for _i in range(_B.shape[0]):
            _bi = _B[_i]
            for _j in range(_i + 1, _B.shape[0]):
                _bj = _B[_j]
                _inter = np.logical_and(_bi, _bj).sum()
                _union = np.logical_or(_bi, _bj).sum()
                _d = 0.0 if _union == 0 else 1.0 - _inter / _union
                _distances.append(_d)

        _distances = np.array(_distances, dtype=float)

        return {
            "pairwise_jaccard_mean": float(np.mean(_distances)),
            "pairwise_jaccard_q25": float(np.quantile(_distances, 0.25)),
            "pairwise_jaccard_median": float(np.median(_distances)),
            "pairwise_jaccard_q75": float(np.quantile(_distances, 0.75)),
        }

    return (v3_pairwise_jaccard_summary,)


@app.cell
def _(
    V3_LATENT_DIM,
    decode_threshold,
    n_prior,
    np,
    torch,
    v3_model,
    v3_pairwise_jaccard_summary,
):
    """Decoder sensitivity: sample prior z values and check decoded diversity."""
    torch.manual_seed(42)

    with torch.no_grad():
        v3_prior_z = torch.randn(n_prior.value, V3_LATENT_DIM)
        v3_prior_probs_np = v3_model.decode(v3_prior_z).cpu().numpy()

    v3_prior_binary_np = (v3_prior_probs_np > decode_threshold.value).astype(np.uint8)
    _prior_gene_counts = v3_prior_binary_np.sum(axis=1)

    _prior_prob_sd_per_gene = v3_prior_probs_np.std(axis=0)

    v3_prior_diversity_summary = {
        "n_prior_samples": int(n_prior.value),
        "decode_threshold": float(decode_threshold.value),
        "genome_size_mean": float(_prior_gene_counts.mean()),
        "genome_size_sd": float(_prior_gene_counts.std()),
        "genome_size_min": int(_prior_gene_counts.min()),
        "genome_size_max": int(_prior_gene_counts.max()),
        "mean_per_gene_probability_sd": float(_prior_prob_sd_per_gene.mean()),
        "median_per_gene_probability_sd": float(np.median(_prior_prob_sd_per_gene)),
        **v3_pairwise_jaccard_summary(v3_prior_binary_np, max_items=300, seed=42),
    }

    print(v3_prior_diversity_summary)
    return v3_prior_binary_np, v3_prior_diversity_summary, v3_prior_probs_np


@app.cell
def _(
    V3_LATENT_DIM,
    X_real_np,
    decode_threshold,
    n_parents,
    n_posterior_per_parent,
    np,
    pd,
    strain_names,
    torch,
    v3_get_mu_logvar,
    v3_model,
    v3_pairwise_jaccard_summary,
):
    """Sample locally from q(z | x_parent) for selected parental genomes."""
    _rng = np.random.default_rng(42)
    _n_parents = min(int(n_parents.value), X_real_np.shape[0])
    _parent_idx = _rng.choice(X_real_np.shape[0], size=_n_parents, replace=False)

    _X = torch.tensor(X_real_np, dtype=torch.float32)

    _posterior_probs = []
    _posterior_rows = []

    with torch.no_grad():
        for _parent_number, _idx in enumerate(_parent_idx):
            _x_parent = _X[_idx : _idx + 1]
            _mu_parent, _logvar_parent = v3_get_mu_logvar(_x_parent)

            _sigma_parent = torch.exp(0.5 * _logvar_parent)
            _eps = torch.randn(int(n_posterior_per_parent.value), V3_LATENT_DIM)
            _z_post = _mu_parent + _sigma_parent * _eps

            _probs_post = v3_model.decode(_z_post).cpu().numpy()
            _posterior_probs.append(_probs_post)

            for _k in range(int(n_posterior_per_parent.value)):
                _posterior_rows.append(
                    {
                        "sample_id": f"posterior_parent_{_parent_number:03d}_{_k:03d}",
                        "parent_number": _parent_number,
                        "parent_strain": strain_names[_idx],
                    }
                )

    v3_posterior_probs_np = np.vstack(_posterior_probs)
    v3_posterior_binary_np = (
        v3_posterior_probs_np > decode_threshold.value
    ).astype(np.uint8)

    _posterior_gene_counts = v3_posterior_binary_np.sum(axis=1)
    _posterior_prob_sd_per_gene = v3_posterior_probs_np.std(axis=0)

    v3_posterior_metadata_df = pd.DataFrame(_posterior_rows)

    v3_posterior_diversity_summary = {
        "n_parents": int(_n_parents),
        "n_posterior_per_parent": int(n_posterior_per_parent.value),
        "n_posterior_samples": int(v3_posterior_binary_np.shape[0]),
        "decode_threshold": float(decode_threshold.value),
        "genome_size_mean": float(_posterior_gene_counts.mean()),
        "genome_size_sd": float(_posterior_gene_counts.std()),
        "genome_size_min": int(_posterior_gene_counts.min()),
        "genome_size_max": int(_posterior_gene_counts.max()),
        "mean_per_gene_probability_sd": float(_posterior_prob_sd_per_gene.mean()),
        "median_per_gene_probability_sd": float(np.median(_posterior_prob_sd_per_gene)),
        **v3_pairwise_jaccard_summary(v3_posterior_binary_np, max_items=300, seed=42),
    }

    print(v3_posterior_diversity_summary)
    return (
        v3_posterior_binary_np,
        v3_posterior_diversity_summary,
        v3_posterior_metadata_df,
        v3_posterior_probs_np,
    )


@app.cell
def _(alt, pd, v3_posterior_binary_np, v3_prior_binary_np):
    _plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "source": "prior",
                    "n_genes": v3_prior_binary_np.sum(axis=1),
                }
            ),
            pd.DataFrame(
                {
                    "source": "posterior-parent",
                    "n_genes": v3_posterior_binary_np.sum(axis=1),
                }
            ),
        ],
        ignore_index=True,
    )

    _chart=alt.Chart(_plot_df).mark_bar(opacity=0.65).encode(
        x=alt.X(
            "n_genes:Q",
            bin=alt.Bin(maxbins=40),
            title="genes per decoded sample",
        ),
        y=alt.Y("count():Q", title="# samples"),
        color=alt.Color("source:N"),
        tooltip=["source", "count()"],
    ).properties(
        width=520,
        height=260,
        title="v3 prior vs posterior-parent genome-size distributions",
    )
    _chart.save("plot_latent_prior_vs_posterior_parent_size.pdf")
    _chart
    return


@app.cell
def _(gene_order, v3_posterior_binary_np):
    def v3_masks_to_gene_lists(binary_matrix, gene_order_array):
        _gene_lists = []
        for _row in binary_matrix:
            _gene_lists.append(gene_order_array[_row.astype(bool)].tolist())
        return _gene_lists

    v3_posterior_gene_lists = v3_masks_to_gene_lists(
        v3_posterior_binary_np,
        gene_order,
    )

    print(f"Posterior gene lists: {len(v3_posterior_gene_lists)}")
    return (v3_posterior_gene_lists,)


@app.cell
def _(
    X_real_np,
    np,
    pd,
    strain_names,
    v3_posterior_binary_np,
    v3_posterior_metadata_df,
    v3_prior_binary_np,
):
    def _rows_to_sets(_B):
        return [set(np.flatnonzero(_row)) for _row in _B]

    def _jaccard_distance(_a, _b):
        _union = len(_a | _b)
        if _union == 0:
            return 0.0
        return 1.0 - len(_a & _b) / _union

    def _summarise(_label, _values):
        _values = np.asarray(_values, dtype=float)
        return {
            "comparison": _label,
            "n_pairs": int(len(_values)),
            "mean": float(np.mean(_values)),
            "q25": float(np.quantile(_values, 0.25)),
            "median": float(np.median(_values)),
            "q75": float(np.quantile(_values, 0.75)),
        }

    def _sample_pairwise(_sets, _n_pairs=20000, _seed=42):
        _rng = np.random.default_rng(_seed)
        _n = len(_sets)
        _values = []

        for _ in range(_n_pairs):
            _i = _rng.integers(0, _n)
            _j = _rng.integers(0, _n - 1)
            if _j >= _i:
                _j += 1
            _values.append(_jaccard_distance(_sets[_i], _sets[_j]))

        return _values

    def _sample_between_labels(_sets, _labels, _n_pairs=20000, _seed=42):
        _rng = np.random.default_rng(_seed)
        _n = len(_sets)
        _values = []

        for _ in range(_n_pairs):
            while True:
                _i = _rng.integers(0, _n)
                _j = _rng.integers(0, _n)
                if _i != _j and _labels[_i] != _labels[_j]:
                    break
            _values.append(_jaccard_distance(_sets[_i], _sets[_j]))

        return _values

    _real_sets = _rows_to_sets(X_real_np.astype(np.uint8))
    _prior_sets = _rows_to_sets(v3_prior_binary_np)
    _posterior_sets = _rows_to_sets(v3_posterior_binary_np)

    _posterior_labels = v3_posterior_metadata_df["parent_number"].to_numpy()

    _rows = []

    # Prior-prior diversity
    _rows.append(
        _summarise(
            "prior-prior",
            _sample_pairwise(_prior_sets, _n_pairs=20000, _seed=1),
        )
    )

    # Real-real reference diversity
    _rows.append(
        _summarise(
            "real-real",
            _sample_pairwise(_real_sets, _n_pairs=20000, _seed=2),
        )
    )

    # Posterior-posterior across all parents
    _rows.append(
        _summarise(
            "posterior-posterior, all parents",
            _sample_pairwise(_posterior_sets, _n_pairs=20000, _seed=3),
        )
    )

    # Posterior-posterior within the same parent
    _within_parent_values = []
    for _parent_number, _group in v3_posterior_metadata_df.groupby("parent_number"):
        _idx = _group.index.to_numpy()
        for _i_pos in range(len(_idx)):
            for _j_pos in range(_i_pos + 1, len(_idx)):
                _i = _idx[_i_pos]
                _j = _idx[_j_pos]
                _within_parent_values.append(
                    _jaccard_distance(_posterior_sets[_i], _posterior_sets[_j])
                )

    _rows.append(
        _summarise(
            "posterior-posterior, same parent",
            _within_parent_values,
        )
    )

    # Posterior-posterior between different parents
    _rows.append(
        _summarise(
            "posterior-posterior, different parents",
            _sample_between_labels(
                _posterior_sets,
                _posterior_labels,
                _n_pairs=20000,
                _seed=4,
            ),
        )
    )

    # Posterior sample to its own parent
    _strain_to_real_idx = {_s: _i for _i, _s in enumerate(strain_names)}
    _to_own_parent_values = []

    for _sample_idx, _row in v3_posterior_metadata_df.iterrows():
        _parent_strain = _row["parent_strain"]
        _parent_idx = _strain_to_real_idx[_parent_strain]
        _to_own_parent_values.append(
            _jaccard_distance(
                _posterior_sets[_sample_idx],
                _real_sets[_parent_idx],
            )
        )

    _rows.append(
        _summarise(
            "posterior-to-own-parent",
            _to_own_parent_values,
        )
    )

    # Posterior sample to a random unrelated real genome
    _rng = np.random.default_rng(5)
    _to_random_real_values = []

    for _sample_idx, _row in v3_posterior_metadata_df.iterrows():
        _own_idx = _strain_to_real_idx[_row["parent_strain"]]

        while True:
            _random_idx = _rng.integers(0, len(_real_sets))
            if _random_idx != _own_idx:
                break

        _to_random_real_values.append(
            _jaccard_distance(
                _posterior_sets[_sample_idx],
                _real_sets[_random_idx],
            )
        )

    _rows.append(
        _summarise(
            "posterior-to-random-real",
            _to_random_real_values,
        )
    )

    v3_jaccard_comparison_df = pd.DataFrame(_rows)

    v3_jaccard_comparison_df
    return (v3_jaccard_comparison_df,)


@app.cell
def _(alt, v3_jaccard_comparison_df):
    _chart = alt.Chart(v3_jaccard_comparison_df).mark_bar().encode(
        x=alt.X(
            "median:Q",
            title="median Jaccard distance",
        ),
        y=alt.Y(
            "comparison:N",
            sort="-x",
            title=None,
        ),
        tooltip=[
            "comparison",
            "n_pairs",
            alt.Tooltip("mean:Q", format=".3f"),
            alt.Tooltip("q25:Q", format=".3f"),
            alt.Tooltip("median:Q", format=".3f"),
            alt.Tooltip("q75:Q", format=".3f"),
        ],
    ).properties(
        width=420,
        height=260,
        title="Jaccard distance decomposition",
    )
    _chart.save("plot_latent_jaccard.pdf")
    _chart
    return


@app.cell
def _(
    OUT_DIR,
    PROJECT_ROOT,
    json,
    np,
    pd,
    v3_active_latent_df,
    v3_active_summary,
    v3_posterior_diversity_summary,
    v3_posterior_gene_lists,
    v3_posterior_metadata_df,
    v3_posterior_probs_np,
    v3_prior_diversity_summary,
    v3_prior_probs_np,
):
    """Write outputs for downstream analysis and reporting."""
    from genome_minimizer_2.explore_data.binary_converter import (
        check_essential_genes,
        load_files,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _active_path = OUT_DIR / "v3_active_latent_components.tsv"
    _summary_path = OUT_DIR / "v3_latent_diagnostic_summary.tsv"
    _summary_json_path = OUT_DIR / "v3_latent_diagnostic_summary.json"
    _prior_probs_path = OUT_DIR / "v3_prior_decoder_probs.npy"
    _posterior_probs_path = OUT_DIR / "v3_posterior_parental_decoder_probs.npy"
    _posterior_gene_lists_path = OUT_DIR / "v3_posterior_parental_gene_lists.npy"
    _posterior_metadata_path = OUT_DIR / "v3_posterior_parental_metadata.tsv"

    v3_active_latent_df.to_csv(_active_path, sep="\t", index=False)
    np.save(_prior_probs_path, v3_prior_probs_np)
    np.save(_posterior_probs_path, v3_posterior_probs_np)
    np.save(
        _posterior_gene_lists_path,
        np.array(v3_posterior_gene_lists, dtype=object),
    )
    v3_posterior_metadata_df.to_csv(_posterior_metadata_path, sep="\t", index=False)

    # Add literature essential genes to posterior-parental gene lists so these can
    # be dropped straight into your existing FBA / BioGRID analysis.
    _essential_csv = PROJECT_ROOT / "data" / "essential_genes.csv"
    _essential_set, _id_lists = load_files(
        str(_essential_csv),
        str(_posterior_gene_lists_path),
    )
    _posterior_filled_path = check_essential_genes(
        _essential_set,
        _id_lists,
        str(_posterior_gene_lists_path),
    )

    v3_latent_summary = {
        **v3_active_summary,
        **{f"prior_{_k}": _v for _k, _v in v3_prior_diversity_summary.items()},
        **{f"posterior_{_k}": _v for _k, _v in v3_posterior_diversity_summary.items()},
        "posterior_gene_lists_with_essentials_path": str(_posterior_filled_path),
    }

    pd.DataFrame([v3_latent_summary]).to_csv(
        _summary_path,
        sep="\t",
        index=False,
    )
    _summary_json_path.write_text(json.dumps(v3_latent_summary, indent=2))

    print("Wrote:")
    print(f"- {_active_path}")
    print(f"- {_summary_path}")
    print(f"- {_summary_json_path}")
    print(f"- {_prior_probs_path}")
    print(f"- {_posterior_probs_path}")
    print(f"- {_posterior_gene_lists_path}")
    print(f"- {_posterior_filled_path}")
    print(f"- {_posterior_metadata_path}")
    return (v3_latent_summary,)


@app.cell
def _(mo, v3_latent_summary):
    mo.md(f"""
    ## Summary for reviewer response

    Active latent dimensions:

    - `n_active_by_kl`: **{v3_latent_summary["n_active_by_kl"]}**
    - `n_active_by_var_mu`: **{v3_latent_summary["n_active_by_var_mu"]}**
    - `n_active_by_either`: **{v3_latent_summary["n_active_by_either"]}**
    - `total_mean_kl_nats`: **{v3_latent_summary["total_mean_kl_nats"]:.3f}**

    Decoder sensitivity from prior samples:

    - mean genome size: **{v3_latent_summary["prior_genome_size_mean"]:.1f}**
    - genome-size SD: **{v3_latent_summary["prior_genome_size_sd"]:.1f}**
    - median pairwise Jaccard distance: **{v3_latent_summary["prior_pairwise_jaccard_median"]:.3f}**

    Posterior-parental samples:

    - mean genome size: **{v3_latent_summary["posterior_genome_size_mean"]:.1f}**
    - genome-size SD: **{v3_latent_summary["posterior_genome_size_sd"]:.1f}**
    - median pairwise Jaccard distance: **{v3_latent_summary["posterior_pairwise_jaccard_median"]:.3f}**

    The posterior-parental gene lists with essentials are written to:

    `{v3_latent_summary["posterior_gene_lists_with_essentials_path"]}`

    You can add this file as another cohort in the existing FBA / BioGRID
    notebook to test whether posterior sampling increases viability or
    reduces co-deleted negative genetic interactions relative to prior
    samples at matched genome size.
    """)
    return


if __name__ == "__main__":
    app.run()
