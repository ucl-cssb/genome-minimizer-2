#!/usr/bin/env python3
"""
Data Exploration and Analysis for Genomics Dataset

This script performs comprehensive exploratory data analysis on the genomics dataset,
including genome size distributions, gene frequency analysis, essential gene analysis,
and phylogroup-based PCA visualization.
"""

# Import libraries
import sys
import os
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, List
import logging

# Setup project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directories first
from ..utils.directories import (
    PROJECT_ROOT, 
    TEN_K_DATASET_FULL, 
    TEN_K_DATASET_PHYLOGROUPS_FULL, 
    PAPER_ESSENTIAL_GENES_FULL
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FIGURES_DIR = Path(PROJECT_ROOT) / "data" / "data_exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory ready: {FIGURES_DIR}")
FIGURE_SIZE = (4, 4)
PLOT_COLOR = "darkorchid"
PLOT_DPI = 300

def clean_gene_name(gene):
    """Clean and validate gene names, filtering out NaN/None values"""
    if pd.isna(gene) or gene is None:
        return None
    gene = str(gene).strip()
    return gene if gene else None


def load_and_validate_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate all required datasets.
    
    Returns:
        Tuple of (main_data, phylogroup_data, merged_data)
    
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data validation fails
    """
    logger.info("Loading datasets...")
    
    try:
        # Load main genomics dataset
        large_data = pd.read_csv(TEN_K_DATASET_FULL, index_col=0, header=0)
        large_data.columns = large_data.columns.str.upper()
        print(large_data)
        
        # Load phylogroup data
        phylogroup_data = pd.read_csv(TEN_K_DATASET_PHYLOGROUPS_FULL, index_col=0, header=0)
        logger.info(f"Phylogroup data loaded: {phylogroup_data.shape}")
        
        # Remove lineage row and transpose for analysis
        data_without_lineage = large_data.drop(index=['Lineage'], errors='ignore')

        logger.info(f"Main dataset loaded: {data_without_lineage.shape} (genes x samples)")
        
        # Merge with phylogroup information
        merged_df = pd.merge(
            data_without_lineage.transpose(), 
            phylogroup_data, 
            how='inner', 
            left_index=True, 
            right_on='ID'
        )
        logger.info(f"Merged dataset: {merged_df.shape} (samples x genes+phylogroup)")
        
        # Validate data integrity
        if merged_df.empty:
            raise ValueError("Merged dataset is empty - check ID matching between datasets")
        
        if 'Phylogroup' not in merged_df.columns:
            raise ValueError("Phylogroup column not found in merged data")
        
        logger.info("✓ Data validation passed")
        return large_data, merged_df, data_without_lineage
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ Required data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Error loading data: {e}")
        raise


def create_genome_size_distribution_plot(data_without_lineage: pd.DataFrame) -> None:
    """
    Create Figure 1a: Distribution of genes across genomes.
    
    Args:
        data_without_lineage: Main dataset without lineage information
    """
    logger.info("Creating genome size distribution plot (Figure 1a)...")
    
    # Calculate frequency: number of genomes each gene appears in
    gene_frequencies = data_without_lineage.sum(axis=0).values
    
    plt.figure(figsize=FIGURE_SIZE, dpi=PLOT_DPI)
    plt.hist(gene_frequencies, color=PLOT_COLOR, bins=20)
    plt.xlabel('Number of Genomes')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Frequencies Across Genomes')

    stats = {
        'median': np.median(gene_frequencies),
        'mean': np.mean(gene_frequencies),
        'min': np.min(gene_frequencies),
        'max': np.max(gene_frequencies),
    }

    plt.axvline(stats['median'], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}")

    # FIX: Use dictionary keys consistently
    dummy_min = plt.Line2D([], [], color='black', linewidth=2, label=f"Min: {stats['min']:.2f}")
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f"Max: {stats['max']:.2f}")

    # FIX: Use dictionary keys in the handles list too
    handles = [
        plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}"), 
        dummy_min, 
        dummy_max
    ]

    plt.legend(handles=handles, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot_genome_size_final.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info("✓ Figure 1a saved")


def create_gene_count_distribution_plot(data_without_lineage: pd.DataFrame) -> None:
    """
    Create Figure 1b: Distribution of genome sizes (genes per genome).
    
    Args:
        data_without_lineage: Main dataset without lineage information
    """
    logger.info("Creating gene count distribution plot (Figure 1b)...")
    
    # Calculate genome sizes: number of genes per genome
    genome_sizes = data_without_lineage.sum(axis=1)
    
    # Calculate statistics
    stats = {
        'mean': np.mean(genome_sizes),
        'median': np.median(genome_sizes),
        'min': np.min(genome_sizes),
        'max': np.max(genome_sizes),
        'std': np.std(genome_sizes)
    }
    
    plt.figure(figsize=FIGURE_SIZE, dpi=PLOT_DPI)
    plt.hist(genome_sizes, color=PLOT_COLOR, bins=20)
    plt.xlabel('Number of Genes')
    plt.ylabel('Number of Genomes')
    plt.title('Distribution of Genome Sizes')
    
    plt.axvline(stats['median'], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}")

    # FIX: Use dictionary keys consistently
    dummy_min = plt.Line2D([], [], color='black', linewidth=2, label=f"Min: {stats['min']:.2f}")
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f"Max: {stats['max']:.2f}")

    # FIX: Use dictionary keys in the handles list too
    handles = [
        plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}"), 
        dummy_min, 
        dummy_max
    ]

    plt.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot_gene_count_final.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info("✓ Figure 1b saved")


def create_gene_frequency_threshold_plot(data_without_lineage: pd.DataFrame) -> None:
    """
    Create Figure 1c: Gene frequency threshold analysis.
    
    Args:
        data_without_lineage: Main dataset without lineage information
    """
    logger.info("Creating gene frequency threshold plot (Figure 1c)...")
    
    # Define thresholds and calculate gene counts
    thresholds = np.linspace(0, 50, num=50)
    threshold_data = []
    
    for threshold in thresholds:
        # Count genes that appear in at least 'threshold' genomes
        gene_frequencies = data_without_lineage.sum(axis=1)
        genes_above_threshold = len(data_without_lineage[gene_frequencies >= threshold])
        threshold_data.append(genes_above_threshold)
    
    plt.figure(figsize=FIGURE_SIZE, dpi=PLOT_DPI)
    plt.scatter(thresholds, threshold_data, color=PLOT_COLOR, alpha=0.7, s=30)
    plt.plot(thresholds, threshold_data, color=PLOT_COLOR, linewidth=2)
    plt.xlabel("Minimum Number of Genomes")
    plt.ylabel("Number of Genes")
    plt.title("Gene Frequency Threshold Analysis")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot_gene_frequency_final.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info("✓ Figure 1c saved")


def process_essential_genes(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process essential genes and match them to dataset genes.
    
    Args:
        merged_df: Merged dataset with phylogroup information
    
    Returns:
        Tuple of (essential_genes_dataframe, essential_gene_list)
    """
    logger.info("Processing essential genes...")
    
    try:
        # Load essential genes list
        essential_genes = pd.read_csv(PAPER_ESSENTIAL_GENES_FULL)
        essential_genes_array = essential_genes.values.flatten()
        logger.info(f"Loaded {len(essential_genes_array)} essential genes from paper")
        
        # Get all genes from dataset (excluding phylogroup column)
        all_genes = merged_df.columns[:-1]  # Exclude 'Phylogroup' column
        
        # Direct matches
        direct_matches = essential_genes[essential_genes.isin(all_genes)]
        present_genes = direct_matches.values.flatten()
        
        # Find genes not directly present
        absent_genes = essential_genes[~essential_genes.isin(all_genes)].values.flatten()
        logger.info(f"Direct matches: {len(present_genes)}, Absent: {len(absent_genes)}")
        
        matched_columns = []
        for gene in absent_genes:
            clean_gene = clean_gene_name(gene)
            if clean_gene is None:
                continue
            
            try:
                pattern = re.compile(f"^{re.escape(clean_gene)}")
                matches = [col for col in all_genes if pattern.match(col) and col not in present_genes]
                matched_columns.extend(matches)
            except Exception as e:
                logger.warning(f"Error processing gene '{clean_gene}': {e}")
                continue
        
        divided_genes = np.array(matched_columns)
        logger.info(f"Prefix matches found: {len(divided_genes)}")
        
        # Known gene prefixes that should be consolidated
        divided_genes_prefixes = [
            'msbA', 'fabG', 'lolD', 'topA', 'metG', 'fbaA', 
            'higA', 'lptB', 'ssb', 'lptG', 'dnaC'
        ]
        
        # Genes still not found
        not_present = set(absent_genes) - set(divided_genes_prefixes)
        if not_present:
            logger.warning(f"Genes still not found: {len(not_present)} genes")
            logger.debug(f"Missing genes: {list(not_present)[:10]}...")  # Log first 10
        
        # Combine all essential gene matches
        combined_array = np.concatenate((present_genes, divided_genes))
        logger.info(f"Total essential genes in dataset: {len(combined_array)}")
        
        # Get all genes from dataset (excluding phylogroup column)
        all_genes = merged_df.columns[:-1]  # This should be length 55039

        # Create mask for just the gene columns
        essential_genes_mask = all_genes.isin(combined_array)  # Length 55039

        # Apply mask to gene columns only
        essential_genes_df = merged_df.iloc[:, :-1].loc[:, essential_genes_mask].copy()  # Apply to genes only
        
        # Check for genes with zero presence
        gene_sums = essential_genes_df.sum()
        zero_sum_genes = gene_sums[gene_sums == 0].index.tolist()
        if zero_sum_genes:
            logger.warning(f"Essential genes with zero presence: {len(zero_sum_genes)}")
        
        # Process absent essential genes by consolidating variants
        absent_essential_genes_df = pd.DataFrame(index=essential_genes_df.index)
        
        for prefix in absent_genes:
            clean_prefix = clean_gene_name(prefix)
            if clean_prefix is None:
                continue
            
            try:
                prefix_cols = essential_genes_df.filter(regex=f'^{re.escape(clean_prefix)}')
                if not prefix_cols.empty:
                    absent_essential_genes_df[clean_prefix] = (prefix_cols.sum(axis=1) > 0).astype(int)
            except Exception as e:
                logger.warning(f"Error processing gene prefix '{clean_prefix}': {e}")
                continue
        
        # Remove individual variants and add consolidated genes
        final_essential_df = essential_genes_df.drop(columns=divided_genes, errors='ignore')
        
        # Add consolidated absent genes that have non-zero presence
        genes_to_add = absent_essential_genes_df.columns[absent_essential_genes_df.sum(axis=0) > 0]
        for gene in genes_to_add:
            final_essential_df[gene] = absent_essential_genes_df[gene]
        
        logger.info(f"Final essential genes dataframe: {final_essential_df.shape}")
        
        # Save essential gene list
        essential_gene_list = final_essential_df.columns.tolist()
        np.save(Path(PROJECT_ROOT) / 'data' / 'essential_genes' / 'essential_gene_in_ds.npy', essential_gene_list)
        logger.info("✓ Essential gene list saved")
        
        return final_essential_df
        
    except Exception as e:
        logger.error(f"\n✗ Error processing essential genes: {e}")
        raise


def create_essential_genes_distribution_plot(essential_genes_df: pd.DataFrame) -> None:
    """
    Create Figure 1d: Distribution of essential gene counts per genome.
    
    Args:
        essential_genes_df: Dataframe containing essential genes data
    """
    logger.info("Creating essential genes distribution plot (Figure 1d)...")
    
    # Calculate essential gene counts per genome
    essential_gene_counts = essential_genes_df.sum(axis=1)
    
    stats = {
        'median': np.median(essential_gene_counts),
        'mean': np.mean(essential_gene_counts),
        'min': np.min(essential_gene_counts),
        'max': np.max(essential_gene_counts),
    }
    
    plt.figure(figsize=FIGURE_SIZE, dpi=PLOT_DPI)
    plt.hist(essential_gene_counts, color=PLOT_COLOR, bins=50)
    
    plt.xlabel('Essential Gene Number')
    plt.ylabel('Frequency')
    plt.title('Distribution of Essential Genes per Genome')
    
    plt.axvline(stats['median'], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}")

    dummy_min = plt.Line2D([], [], color='black', linewidth=2, label=f"Min: {stats['min']:.2f}")
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f"Max: {stats['max']:.2f}")

    handles = [
        plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}"), 
        dummy_min, 
        dummy_max
    ]
    plt.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot_EG_number.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info("✓ Figure 1d saved")


def create_pca_phylogroup_plot(merged_df: pd.DataFrame) -> None:
    """Create Figure 2a: PCA visualization colored by phylogroup."""
    logger.info("Creating PCA by phylogroup plot (Figure 2a)...")
    
    gene_data = merged_df.iloc[:, :-1].values
    phylogroups = merged_df['Phylogroup'].values
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(gene_data)
    
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    df_pca['Phylogroup'] = phylogroups
    
    plt.figure(figsize=FIGURE_SIZE, dpi=PLOT_DPI)
    
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Phylogroup', alpha=0.7, s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA Analysis by Phylogroup')
    
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "plot_PCA_by_phylogroup.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info("✓ Figure 2a saved")

def generate_summary_report(
    merged_df: pd.DataFrame, 
    essential_genes_df: pd.DataFrame
) -> None:
    """
    Generate a summary report of the data exploration analysis.
    
    Args:
        large_data: Original dataset
        merged_df: Merged dataset with phylogroup information  
        essential_genes_df: Essential genes dataframe
    """
    logger.info("Generating summary report...")
    
    # Calculate summary statistics
    n_genomes = merged_df.shape[0]
    n_genes = merged_df.shape[1] - 1  # Exclude phylogroup column
    n_essential_genes = essential_genes_df.shape[1]
    
    genome_sizes = merged_df.iloc[:, :-1].sum(axis=1)
    essential_counts = essential_genes_df.sum(axis=1)
    
    phylogroup_counts = merged_df['Phylogroup'].value_counts()
    
    report = f"""
    ===============================================
    GENOMICS DATA EXPLORATION SUMMARY REPORT
    ===============================================
    
    Dataset Overview:
    - Total genomes: {n_genomes:,}
    - Total genes: {n_genes:,} 
    - Essential genes identified: {n_essential_genes:,}
    - Phylogroups: {len(phylogroup_counts)}
    
    Genome Size Statistics:
    - Mean genome size: {genome_sizes.mean():.0f} genes
    - Median genome size: {genome_sizes.median():.0f} genes
    - Range: {genome_sizes.min():.0f} - {genome_sizes.max():.0f} genes
    - Standard deviation: {genome_sizes.std():.0f} genes
    
    Essential Genes Statistics:
    - Mean essential genes per genome: {essential_counts.mean():.1f}
    - Median essential genes per genome: {essential_counts.median():.0f}
    - Range: {essential_counts.min():.0f} - {essential_counts.max():.0f}
    - Standard deviation: {essential_counts.std():.1f}
    
    Phylogroup Distribution:
    """
    
    for phylogroup, count in phylogroup_counts.items():
        percentage = (count / n_genomes) * 100
        report += f"    - {phylogroup}: {count:,} genomes ({percentage:.1f}%)\n"
    
    report += f"""
    Generated Figures:
    - Figure 1a: Gene frequency distribution (plot_genome_size_final.pdf)
    - Figure 1b: Genome size distribution (plot_gene_count_final.pdf)  
    - Figure 1c: Gene frequency thresholds (plot_gene_frequency_final.pdf)
    - Figure 1d: Essential genes distribution (plot_EG_number.pdf)
    - Figure 2a: PCA by phylogroup (plot_PCA_by_phylogroup.pdf)
    
    Output Directory: {FIGURES_DIR}
    ===============================================
    """
    
    # Save report to file
    report_file = FIGURES_DIR / "data_exploration_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Also print to console
    print(report)
    logger.info(f"✓ Summary report saved to: {report_file}")


def main():
    """Main execution function."""
    logger.info("Starting data exploration analysis...")
    
    try:     
        large_data, merged_df, data_without_lineage = load_and_validate_data()
        
        # Generate Figure 1 series - Dataset overview
        create_genome_size_distribution_plot(data_without_lineage)
        create_gene_count_distribution_plot(data_without_lineage)
        create_gene_frequency_threshold_plot(data_without_lineage)
        
        # Process essential genes
        essential_genes_df = process_essential_genes(merged_df)
        
        # Generate Figure 1d - Essential genes
        create_essential_genes_distribution_plot(essential_genes_df)
        
        # Generate Figure 2a - PCA analysis
        create_pca_phylogroup_plot(merged_df)
        
        # Generate summary report
        generate_summary_report(merged_df, essential_genes_df)
        
        logger.info("\n✓ DATA EXPLORATION COMPLETED!")
        logger.info(f"- All figures saved to: {FIGURES_DIR}")
        
    except Exception as e:
        logger.error(f"\n✗ Data exploration failed: {e}")
        raise


if __name__ == "__main__":
    main()