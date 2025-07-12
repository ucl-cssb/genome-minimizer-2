#!/usr/bin/env python3
"""
Essential Genes Position Retrieval and Processing
"""

# Import libraries
import sys
import os
import re
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import shared data loading function
from ..explore_data.data_exploration import load_and_validate_data

# Import other utilities
from ..utils.extras import extract_prefix
from ..utils.directories import (
    PAPER_ESSENTIAL_GENES_FULL,
    ESSENTIAL_GENES_POSITIONS,
    PROJECT_ROOT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
KNOWN_GENE_PREFIXES = [
    'msbA', 'fabG', 'lolD', 'topA', 'metG', 'fbaA', 
    'higA', 'lptB', 'ssb', 'lptG', 'dnaC'
]


class EssentialGeneProcessor:
    """
    Processes essential genes and maps them to dataset positions.
    
    This class handles the complex task of matching essential gene names
    from literature to actual gene names in the dataset, accounting for
    naming variations and gene families.
    """
    
    def __init__(self):
        """Initialize the processor"""
        self.large_data = None
        self.merged_df = None
        self.data_without_lineage = None
        self.essential_genes_array = None
        self.all_genes = None
        self.gene_position_mapping = {}

        self.figure_dir = Path(PROJECT_ROOT) / "data" / "essential_genes" 
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self) -> None:
        """Load and prepare all required datasets using shared data loading function."""
        logger.info("Loading datasets using shared data loading function...")
        
        try:
            # Use the shared data loading function from data exploration
            self.large_data, self.merged_df, self.data_without_lineage = load_and_validate_data()
            
            logger.info(f"Datasets loaded successfully:")
            logger.info(f"- Main dataset: {self.large_data.shape}")
            logger.info(f"- Merged dataset: {self.merged_df.shape}")
            logger.info(f"- Data without lineage: {self.data_without_lineage.shape}")
            
            # Get all gene names (excluding phylogroup column)
            self.all_genes = self.merged_df.columns[:-1]
            logger.info(f"Total genes in dataset: {len(self.all_genes)}")
            
            # Load essential genes list
            essential_genes_df = pd.read_csv(PAPER_ESSENTIAL_GENES_FULL)
            self.essential_genes_array = essential_genes_df.values.flatten()
            logger.info(f"Essential genes from literature: {len(self.essential_genes_array)}")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def create_gene_position_mapping(self) -> Dict[str, List[int]]:
        """
        Create mapping from gene prefixes to their positions in the dataset.
        
        Returns:
            Dictionary mapping gene prefixes to lists of column positions
        """
        logger.info("Creating gene position mapping...")
        
        gene_positions = defaultdict(list)
        
        for idx, gene in enumerate(self.all_genes):
            prefix = extract_prefix(gene)
            gene_positions[prefix].append(idx)
        
        # Convert to regular dict
        self.gene_position_mapping = dict(gene_positions)
        
        logger.info(f"Mapped {len(self.gene_position_mapping)} unique gene prefixes")
        return self.gene_position_mapping
    
    def identify_gene_matches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify direct matches and absent essential genes.
        
        Returns:
            Tuple of (present_genes, absent_genes, matched_variant_genes)
        """
        logger.info("Identifying gene matches...")
        
        # Find direct matches
        direct_mask = pd.Series(self.essential_genes_array).isin(self.all_genes)
        present_genes = self.essential_genes_array[direct_mask]
        absent_genes = self.essential_genes_array[~direct_mask]
        
        logger.info(f"Direct matches: {len(present_genes)}")
        logger.info(f"Absent genes: {len(absent_genes)}")
        
        # Find variant matches for absent genes
        matched_variants = []
        
        for gene in absent_genes:
            # Use regex to find genes that start with the essential gene name
            try:
                pattern = re.compile(f"^{re.escape(gene)}")
                matches = [col for col in self.all_genes 
                          if pattern.match(col) and col not in present_genes]
                matched_variants.extend(matches)
            except Exception as e:
                logger.warning(f"Error processing gene '{gene}': {e}")
                continue
        
        matched_variant_genes = np.array(matched_variants)
        logger.info(f"Variant matches found: {len(matched_variant_genes)}")
        
        return present_genes, absent_genes, matched_variant_genes
    
    def consolidate_gene_families(self, absent_genes: np.ndarray) -> pd.DataFrame:
        """
        Consolidate gene family variants into single binary indicators.
        
        Args:
            absent_genes: Array of essential genes not directly found
            
        Returns:
            DataFrame with consolidated gene family presence/absence
        """
        logger.info("Consolidating gene families...")
        
        # Get essential genes subset from merged data
        all_essential_matches = np.concatenate([
            self.identify_gene_matches()[0],  # present_genes
            self.identify_gene_matches()[2]   # matched_variant_genes
        ])
        
        essential_mask = self.merged_df.columns[:-1].isin(all_essential_matches)
        essential_genes_df = self.merged_df.loc[:, essential_mask].copy()
        
        # Create consolidated dataframe for absent gene families
        consolidated_df = pd.DataFrame(index=essential_genes_df.index)
        
        for gene_family in absent_genes:
            try:
                # Find all variants of this gene family
                family_pattern = f"^{re.escape(gene_family)}"
                family_variants = essential_genes_df.filter(regex=family_pattern)
                
                if not family_variants.empty:
                    # Mark as present if ANY variant is present
                    consolidated_df[gene_family] = (family_variants.sum(axis=1) > 0).astype(int)
                    logger.debug(f"Consolidated {len(family_variants.columns)} variants for {gene_family}")
            except Exception as e:
                logger.warning(f"Error consolidating gene family '{gene_family}': {e}")
                continue
        
        logger.info(f"Consolidated {len(consolidated_df.columns)} gene families")
        return consolidated_df
    
    def create_final_essential_genes_mapping(self) -> Dict[str, List[int]]:
        """
        Create the final mapping of essential genes to their positions.
        
        Returns:
            Dictionary mapping essential gene names to their column positions
        """
        logger.info("Creating final essential genes position mapping...")
        
        present_genes, absent_genes, matched_variant_genes = self.identify_gene_matches()
        
        # Create the essential gene positions dictionary
        essential_gene_positions = {}
        
        # Add direct matches
        for gene in present_genes:
            if gene in self.gene_position_mapping:
                essential_gene_positions[gene] = self.gene_position_mapping[gene]
        
        # Add gene families that have been found in dataset
        for gene_family in absent_genes:
            if gene_family in self.gene_position_mapping:
                essential_gene_positions[gene_family] = self.gene_position_mapping[gene_family]
        
        logger.info(f"Final essential gene mapping: {len(essential_gene_positions)} genes")
        
        # Log some statistics
        total_positions = sum(len(positions) for positions in essential_gene_positions.values())
        single_position_genes = sum(1 for positions in essential_gene_positions.values() if len(positions) == 1)
        multi_position_genes = len(essential_gene_positions) - single_position_genes
        
        logger.info(f"Total positions mapped: {total_positions}")
        logger.info(f"Single-position genes: {single_position_genes}")
        logger.info(f"Multi-position genes: {multi_position_genes}")
        
        return essential_gene_positions
    
    def validate_essential_genes_mapping(self, essential_positions: Dict[str, List[int]]) -> bool:
        """
        Validate the essential genes mapping for consistency.
        
        Args:
            essential_positions: Dictionary of essential gene positions
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating essential genes mapping...")
        
        try:
            # Check that all positions are valid
            max_position = len(self.all_genes) - 1
            invalid_positions = []
            
            for gene, positions in essential_positions.items():
                for pos in positions:
                    if pos < 0 or pos > max_position:
                        invalid_positions.append((gene, pos))
            
            if invalid_positions:
                logger.error(f"Invalid positions found: {invalid_positions[:5]}...")
                return False
            
            # Check for reasonable coverage
            total_essential_genes = len(self.essential_genes_array)
            mapped_genes = len(essential_positions)
            coverage_ratio = mapped_genes / total_essential_genes
            
            logger.info(f"Essential gene coverage: {mapped_genes}/{total_essential_genes} ({coverage_ratio:.1%})")
            
            if coverage_ratio < 0.5:
                logger.warning("Low essential gene coverage - check gene name matching")
            
            # Check that we haven't mapped too many positions
            total_positions = sum(len(positions) for positions in essential_positions.values())
            if total_positions > len(self.all_genes):
                logger.error("More essential gene positions than total genes - check for duplicates")
                return False
            
            logger.info("Essential genes mapping validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def save_essential_genes_mapping(self, essential_positions: Dict[str, List[int]]) -> None:
        """
        Save the essential genes mapping to a pickle file.
        
        Args:
            essential_positions: Dictionary of essential gene positions
        """
        logger.info("Saving essential genes mapping...")

        pickle_file = self.figure_dir / "essential_gene_positions.pkl"
        
        try:   
            # Save the mapping
            with open(pickle_file, "wb") as f:
                pickle.dump(essential_positions, f)
            
            logger.info(f"Essential gene positions saved to: {pickle_file}")
            
            # Save a human-readable summary
            summary_path = self.figure_dir / "essential_gene_positions_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("Essential Gene Positions Summary\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total essential genes mapped: {len(essential_positions)}\n")
                f.write(f"Total positions: {sum(len(pos) for pos in essential_positions.values())}\n\n")
                
                f.write("Gene Mappings:\n")
                f.write("=" * 80 + "\n")
                for gene, positions in sorted(essential_positions.items()):
                    if len(positions) == 1:
                        f.write(f"{gene}: position {positions[0]}\n")
                    else:
                        f.write(f"{gene}: positions {positions}\n")
            
            logger.info(f"Summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving essential genes mapping: {e}")
            raise
    
    def process(self) -> Dict[str, List[int]]:
        """
        Main processing method that orchestrates the entire workflow.
        
        Returns:
            Dictionary mapping essential genes to their positions
        """
        logger.info("Starting essential genes processing...")
        
        try:
            # Load data using shared function
            self.load_datasets()
            
            # Create gene position mapping
            self.create_gene_position_mapping()
            
            # Create essential genes mapping
            essential_positions = self.create_final_essential_genes_mapping()
            
            # Validate the mapping
            if not self.validate_essential_genes_mapping(essential_positions):
                raise ValueError("Essential genes mapping validation failed")
            
            self.save_essential_genes_mapping(essential_positions)
            
            logger.info("✓ Essential genes processing completed successfully!")
            return essential_positions
            
        except Exception as e:
            logger.error(f"Essential genes processing failed: {e}")
            raise


def print_processing_summary(essential_positions: Dict[str, List[int]]) -> None:
    """
    Print a summary of the processing results.
    
    Args:
        essential_positions: Dictionary of essential gene positions
    """
    print("\n" + "=" * 80)
    print("ESSENTIAL GENES PROCESSING SUMMARY")
    print("=" * 80)
    
    total_genes = len(essential_positions)
    total_positions = sum(len(positions) for positions in essential_positions.values())
    
    single_position = sum(1 for pos in essential_positions.values() if len(pos) == 1)
    multi_position = total_genes - single_position
    
    print(f"Processing Results:")
    print(f"- Essential genes mapped: {total_genes}")
    print(f"- Total dataset positions: {total_positions}")
    print(f"- Single-position genes: {single_position}")
    print(f"- Multi-position genes: {multi_position}")
    
    if multi_position > 0:
        print(f"\nMulti-position genes (gene families):")
        multi_pos_genes = [(gene, len(pos)) for gene, pos in essential_positions.items() if len(pos) > 1]
        multi_pos_genes.sort(key=lambda x: x[1], reverse=True)
        
        for gene, count in multi_pos_genes[:10]:  # Show top 10
            print(f"- {gene}: {count} positions")
        
        if len(multi_pos_genes) > 10:
            print(f"- ... and {len(multi_pos_genes) - 10} more")
    
    print(f"\n- Output saved to: {ESSENTIAL_GENES_POSITIONS}")
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    logger.info("Starting essential genes retrieval and processing...")
    
    try:
        processor = EssentialGeneProcessor()
        essential_positions = processor.process()
        print_processing_summary(essential_positions)
        
        return essential_positions
        
    except Exception as e:
        logger.error(f"Essential genes processing failed: {e}")
        raise


if __name__ == "__main__":
    main()