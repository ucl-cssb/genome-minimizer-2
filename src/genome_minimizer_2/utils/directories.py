#!/usr/bin/env python3
"""
Directory and path configuration
"""

# Import library
import os

# Get project root directory - go up 3 levels from src/genome_minimizer_2/utils/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Data file paths - all now in data folder
TEN_K_DATASET = "data/F4_complete_presence_absence.csv"
TEN_K_DATASET_PHYLOGROUPS = "data/accessionID_phylogroup_BD.csv"
PAPER_ESSENTIAL_GENES = "data/essential_genes.csv"
WILD_TYPE_SEQUENCE = "data/wild_type_sequence.gb"
SAMPLES_BINARY = "data/data_full_validated.npy"

# Generated data paths
ESSENTIAL_GENES_POSITIONS = os.path.join(PROJECT_ROOT, "src", "genome_minimizer_2", "data", "essential_genes", "essential_gene_positions.pkl")
MINIMIZED_GENOME = os.path.join(PROJECT_ROOT, "data", "minimized_genome.fasta")

# Full paths (for backward compatibility)
def get_full_path(relative_path: str) -> str:
    """Convert relative path to full path from project root."""
    return os.path.join(PROJECT_ROOT, relative_path)

# Export commonly used full paths
TEN_K_DATASET_FULL = get_full_path(TEN_K_DATASET)
TEN_K_DATASET_PHYLOGROUPS_FULL = get_full_path(TEN_K_DATASET_PHYLOGROUPS)
PAPER_ESSENTIAL_GENES_FULL = get_full_path(PAPER_ESSENTIAL_GENES)
WILD_TYPE_SEQUENCE_FULL = get_full_path(WILD_TYPE_SEQUENCE)
SAMPLES_BINARY_FULL = get_full_path(SAMPLES_BINARY)