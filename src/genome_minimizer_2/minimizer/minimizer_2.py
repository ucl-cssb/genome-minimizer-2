"""
GENOME MINIMIZER ALGORITHM
"""

# Import libraries
import os
from ..utils.directories import *
import logging
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

from src.genome_minimizer_2.utils.directories import (
    PROJECT_ROOT
)

class GenomeMinimiser:
    def __init__(self,
                 record_path: str = None,
                 needed_genes_path: str = None,
                 idx: int = 0,
                 model_name: str = "",
                 record: SeqRecord = None,
                 all_needed_gene_lists: list = None,
                 needed_genes_list: list = None):
        self.idx = idx
        self.model_name = model_name

        # genome
        self.record = record if record is not None else self.load_genome(record_path)

        self.wildtype_sequence = self.record
        self.original_genome_length = len(self.record.seq)

        # genes for this sample
        if needed_genes_list is not None:
            self.needed_genes = needed_genes_list
        elif all_needed_gene_lists is not None:
            self.needed_genes = all_needed_gene_lists[idx]
        else:
            self.needed_genes = self.get_needed_genes(needed_genes_path)[idx]

        # compute
        self.features = self._extract_non_essential_genes()
        self.positions_to_remove = self._get_positions_to_remove()
        self.reduced_genome_str = self._create_minimized_sequence()
    
    def _extract_non_essential_genes(self) -> list:
        '''
        Extracts non-essential genes from the genome sequence

        Returns:
        non_essential_features (list): A list of non-essential gene features
        '''
        non_essential_features = []

        for feature in self.record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene", [""])[0]
                if gene_name not in self.needed_genes:
                    non_essential_features.append(feature)

        logging.debug(f"Non-essential genes have been found in sequence no. {self.idx}.")
        return non_essential_features

    def _get_positions_to_remove(self) -> set:
        '''
        Gets the positions of non-essential genes to remove from the genome

        Returns:
        positions_to_remove (set): a set of unique positions to remove
        '''
        positions_to_remove = set()

        for feature in self.features:
            start_position = int(feature.location.start)
            end_position = int(feature.location.end)
            positions_to_remove.update(range(start_position, end_position))

        logging.debug(f"BP positions to remove have been found in sequence no. {self.idx}.")
        return positions_to_remove

    def _create_minimized_sequence(self) -> str:
        '''
        Creates a minimized genome sequence by removing non-essential genes

        Returns:
        reduced_genome_str (str): the minimized genome sequence
        '''
        reduced_genome = []

        for i, base in enumerate(self.record.seq):
            if i not in self.positions_to_remove:
                reduced_genome.append(base)
        reduced_genome_str = ''.join(reduced_genome)

        logging.debug(f"Minimised sequence for sequence no. {self.idx} has been created.")
        # print(f"length pg the genomes {len(reduced_genome_str)}")
        return reduced_genome_str

    def save_minimized_genome(self, file_path: str):
        '''
        Saves the minimized genome sequence to a file

        Parameters:
        file_path (str): the file path where to save the reduced sequence

        Returns:
        None
        '''
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.join(PROJECT_ROOT, "minimized_genomes")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(file_path, "w") as output_file:
                output_file.write(f">Minimized_E_coli_K12_MG1655_{self.idx+1}\n")
                output_file.write(str(self.reduced_genome_str))
                logging.info(f"Successfully saved reduced genome: {file_path}")

        except IOError as e:
            logging.error(f"\n✗ Could not write to file: {file_path} - {e}")
            raise

    def load_genome(self, file_path: str) -> SeqRecord:
        '''
        Loads the GenBank genome file

        Parameters:
        file_path (str): path to the GenBank file of the genome

        Returns:
        wildtype_sequence (SeqRecord): A SeqRecord object containing the genome sequence and annotations
        '''
        try:
            logging.info(f"Attempting to load wildtype genome from {file_path}")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            
            if not file_path.endswith((".gb", ".genbank", ".gbff")):
                raise ValueError(f"The file {file_path} could not be read.\nEnsure the file holds a GenBank format.")
            
            wildtype_sequence = SeqIO.read(file_path, "genbank")
            logging.info(f"✓ Successfully loaded genome from {file_path}")
            return wildtype_sequence
        
        except FileNotFoundError as fnf_error:
            logging.error(f"\n✗ Error: {fnf_error}")
            raise
        except ValueError as val_error:
            logging.error(f"\n✗ Error: {val_error}")
            raise
        except Exception as e:
            logging.error(f"\n✗ Error: {e}.\nAn unexpected error occurred.")
            raise

    def get_needed_genes(self, file_path: str) -> list:
        '''
        Loads the numpy file which contains list of lists of the present genes in sequences

        Parameters:
        file_path (str): path to the numpy file containing the genes

        Returns:
        present_genes (list): a list of lists containing the needed genes
        '''
        try:
            logging.info(f"Loading genes file: {os.path.basename(file_path)}")
            
            # File validation
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            
            if not file_path.endswith(".npy"):
                raise ValueError(f"Invalid file format. Expected .npy file, got: {os.path.splitext(file_path)[1]}")
            
            file_size_mb = os.path.getsize(file_path) / (1024**2)
            if file_size_mb > 10:  
                logging.info(f"File size: {file_size_mb:.2f} MB")
            
            # Load the numpy array
            logging.info(f"Loading gene data from numpy array...")
            genes_array = np.load(file_path, allow_pickle=True)
            
            # Get array info
            total_samples = len(genes_array)
            
            # Convert to list
            logging.info(f"Converting {total_samples} samples to list format...")
            present_genes = genes_array.tolist()
            
            sample_genes_count = len(present_genes[self.idx])
            logging.info(f"✓ Successfully loaded {self.idx+1}th sample ({sample_genes_count} genes per sample)")
            
            return present_genes
            
        except FileNotFoundError as fnf_error:
            logging.error(f"✗ File not found: {fnf_error}")
            raise
        except ValueError as val_error:
            logging.error(f"✗ Invalid file format: {val_error}")
            raise
        except np.core._exceptions._ArrayMemoryError:
            logging.error(f"✗ Not enough memory to load file: {file_path}")
            raise MemoryError(f"Insufficient memory to load {file_path}")
        except Exception as e:
            logging.error(f"✗ Unexpected error loading genes: {e}")
            raise

    def plot(self):
        '''
        Plots the distribution of minimized genome sizes
        '''
        if len(self.minimised_genomes_sizes) >= 100:
            print("Plotting reduced genomes size distribution graph...")
            median = np.median(self.minimised_genomes_sizes)
            min_value = np.min(self.minimised_genomes_sizes)
            max_value = np.max(self.minimised_genomes_sizes)
            
            plt.figure(figsize=(4, 4))
            plt.hist(self.minimised_genomes_sizes, bins=10, color="dodgerblue")
            plt.xlabel("Genome size (Mbp)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Minimized Genome Sizes")
            
            plt.axvline(median, color="b", linestyle="dashed", linewidth=2, label=f"Median: {median:.2f}")
            
            # Create dummy lines for legend
            dummy_min = plt.Line2D([], [], color="black", linewidth=2, label=f"Min: {min_value:.2f}")
            dummy_max = plt.Line2D([], [], color="black", linewidth=2, label=f"Max: {max_value:.2f}")
            
            handles = [
                plt.Line2D([], [], color="b", linestyle="dashed", linewidth=2, label=f"Median: {median:.2f}"),
                dummy_min, 
                dummy_max
            ]
            plt.legend(handles=handles)
            
            # Create figures directory if it doesn't exist
            output_dir = os.path.join(PROJECT_ROOT, "minimized_genomes")
            os.makedirs(output_dir, exist_ok=True)
            
            plt.savefig(
                os.path.join(output_dir, f"minimised_genomes_distribution_{self.model_name}.pdf"), 
                format="pdf", 
                bbox_inches="tight"
            )
            plt.close()  # Close the figure to free memory
        else:
            print(f"Not enough data points ({len(self.minimised_genomes_sizes)}) to create meaningful plot. Need at least 100.")

    def get_reduction_stats(self) -> dict:
        '''
        Returns statistics about the genome reduction
        
        Returns:
        dict: Dictionary containing reduction statistics
        '''
        reduced_length = len(self.reduced_genome_str)
        reduction_percentage = ((self.original_genome_length - reduced_length) / self.original_genome_length) * 100
        
        return {
            'original_length': self.original_genome_length,
            'reduced_length': reduced_length,
            'reduction_percentage': reduction_percentage,
            'genes_removed': len(self.features),
            'positions_removed': len(self.positions_to_remove)
        }


def check_sequence_duplicates(sequences_dict: dict) -> dict:
    """
    Check for identical sequences and return duplicate statistics
    
    Parameters:
    sequences_dict (dict): Dictionary mapping sequence_id -> sequence_string
    
    Returns:
    dict: Statistics about duplicates
    """
    sequence_groups = defaultdict(list)
    
    # Group sequences by their content
    for seq_id, sequence in sequences_dict.items():
        sequence_groups[sequence].append(seq_id)
    
    # Find duplicates
    duplicates = {seq: ids for seq, ids in sequence_groups.items() if len(ids) > 1}
    unique_sequences = {seq: ids for seq, ids in sequence_groups.items() if len(ids) == 1}
    
    stats = {
        'total_sequences': len(sequences_dict),
        'unique_sequences': len(sequence_groups),
        'duplicate_groups': len(duplicates),
        'duplicated_sequences': sum(len(ids) for ids in duplicates.values()),
        'unique_only_sequences': len(unique_sequences),
        'duplicates_detail': duplicates,
        'compression_ratio': len(sequence_groups) / len(sequences_dict) if sequences_dict else 0
    }
    
    return stats


def print_duplicate_statistics(duplicate_stats: dict):
    """
    Print comprehensive duplicate sequence statistics
    
    Parameters:
    duplicate_stats (dict): Statistics from check_sequence_duplicates
    """
    print("\n" + "="*80)
    print("SEQUENCE DUPLICATION ANALYSIS")
    print("="*80)
    
    print(f" Overview:")
    print(f"- Total sequences generated: {duplicate_stats['total_sequences']:,}")
    print(f"- Unique sequences: {duplicate_stats['unique_sequences']:,}")
    print(f"- Duplicate groups: {duplicate_stats['duplicate_groups']:,}")
    print(f"- Sequences with duplicates: {duplicate_stats['duplicated_sequences']:,}")
    print(f"- Truly unique sequences: {duplicate_stats['unique_only_sequences']:,}")
    print(f"- Percentage of unique sequences: {duplicate_stats['compression_ratio']:.2%}")
    
    if duplicate_stats['duplicate_groups'] > 0:
        print(f"\n Duplicate Details:")
        duplicates = duplicate_stats['duplicates_detail']
        
        # Sort by number of duplicates
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (sequence, ids) in enumerate(sorted_duplicates[:10]):  # Show top 10
            print(f"Group {i+1}: {len(ids)} identical sequences")
            print(f"- Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            print(f"- IDs: {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")
            print()
        
        if len(sorted_duplicates) > 10:
            print(f"  ... and {len(sorted_duplicates) - 10} more duplicate groups")
    else:
        print(f"\n✓ No duplicate sequences found!")
    
    print("="*80)


def generate_summary_file(output_file: str, model_name: str, genome_path: str, genes_path: str, 
                         original_length: int, minimised_sizes: list, 
                         duplicate_stats: dict):
    '''
    Generate comprehensive summary text file
    '''
    try:
        output_dir = os.path.join(PROJECT_ROOT, "minimized_genomes")
        os.makedirs(output_dir, exist_ok=True)
        summary_filename = os.path.basename(output_file).replace('.fasta', '_summary.txt')
        summary_file = os.path.join(output_dir, summary_filename)
        
        logging.info(f"Generating summary file: {os.path.basename(summary_file)}")
        
        # Calculate statistics
        total_processed = len(minimised_sizes)
        mean_size_mbp = np.mean(minimised_sizes) if minimised_sizes else 0
        median_size_mbp = np.median(minimised_sizes) if minimised_sizes else 0
        min_size_mbp = np.min(minimised_sizes) if minimised_sizes else 0
        max_size_mbp = np.max(minimised_sizes) if minimised_sizes else 0
        std_size_mbp = np.std(minimised_sizes) if minimised_sizes else 0
        
        with open(summary_file, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("GENOME MINIMIZATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Generation info
            f.write("GENERATION INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Generated on: {np.datetime64('now')}\n")
            f.write(f"Output FASTA file: {os.path.basename(output_file)}\n")
            f.write(f"Summary file: {os.path.basename(summary_file)}\n\n")
            
            # Input files
            f.write("INPUT FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Genome template: {os.path.basename(genome_path)}\n")
            f.write(f"Gene lists file: {os.path.basename(genes_path)}\n")
            f.write(f"Original genome length: {original_length:,} bp\n\n")
            
            # Processing statistics
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Successfully processed: {total_processed:,}\n\n")
            
            # Size statistics
            f.write("MINIMIZED GENOME SIZE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean size: {mean_size_mbp:.3f} Mbp ({mean_size_mbp*1e6:,.0f} bp)\n")
            f.write(f"Median size: {median_size_mbp:.3f} Mbp ({median_size_mbp*1e6:,.0f} bp)\n")
            f.write(f"Minimum size: {min_size_mbp:.3f} Mbp ({min_size_mbp*1e6:,.0f} bp)\n")
            f.write(f"Maximum size: {max_size_mbp:.3f} Mbp ({max_size_mbp*1e6:,.0f} bp)\n")
            f.write(f"Standard deviation: {std_size_mbp:.3f} Mbp\n")
            f.write(f"Size range: {max_size_mbp - min_size_mbp:.3f} Mbp\n\n")
            
            # Reduction statistics
            if original_length > 0:
                mean_reduction = ((original_length - mean_size_mbp*1e6) / original_length) * 100
                min_reduction = ((original_length - max_size_mbp*1e6) / original_length) * 100
                max_reduction = ((original_length - min_size_mbp*1e6) / original_length) * 100
                
                f.write("GENOME REDUCTION STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean reduction: {mean_reduction:.2f}%\n")
                f.write(f"Minimum reduction: {min_reduction:.2f}% (largest genome)\n")
                f.write(f"Maximum reduction: {max_reduction:.2f}% (smallest genome)\n\n")
            
            # Duplicate analysis
            f.write("SEQUENCE DUPLICATION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total sequences: {duplicate_stats['total_sequences']:,}\n")
            f.write(f"Unique sequences: {duplicate_stats['unique_sequences']:,}\n")
            f.write(f"Duplicate groups: {duplicate_stats['duplicate_groups']:,}\n")
            f.write(f"Sequences with duplicates: {duplicate_stats['duplicated_sequences']:,}\n")
            f.write(f"Uniqueness ratio: {duplicate_stats['compression_ratio']:.2%}\n")
            
            # Size distribution
            if minimised_sizes:
                f.write(f"\nSIZE DISTRIBUTION SUMMARY\n")
                f.write("-" * 40 + "\n")
                
                # Create size bins
                size_bins = np.linspace(min_size_mbp, max_size_mbp, 6)
                hist, _ = np.histogram(minimised_sizes, bins=size_bins)
                
                for i in range(len(hist)):
                    bin_start = size_bins[i]
                    bin_end = size_bins[i+1]
                    count = hist[i]
                    percentage = (count / len(minimised_sizes)) * 100
                    f.write(f"{bin_start:.2f} - {bin_end:.2f} Mbp: {count:,} genomes ({percentage:.1f}%)\n")
        
        logging.info(f"✓ Summary file saved: {summary_file}")
        
    except Exception as e:
        logging.error(f"✗ Failed to generate summary file: {e}")


def process_multiple_genomes_single_file(genome_path: str, genes_path: str, model_name: str, output_file: str = None):
    """
    Minimize genomes for all lists and save ALL in one FASTA file.
    """
    if not output_file:
        output_file = os.path.join(PROJECT_ROOT, "minimized_genomes", f"minimized_genomes_{model_name}.fasta")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    record = SeqIO.read(genome_path, "genbank")
    all_lists = np.load(genes_path, allow_pickle=True).tolist()
    original_length = len(record.seq)

    sizes_mbp = []
    tot_red_pct = 0.0
    total_length_bp = 0
    genome_number = len(all_lists)

    with open(output_file, "w") as out:
        out.write(f"# Minimized genomes generated using model: {model_name}\n")
        out.write(f"# Total genomes: {genome_number}\n")
        out.write(f"# Generated on: {np.datetime64('now')}\n")

        for idx, needed in enumerate(all_lists):
            sample_genes_count = len(needed)
            print(f"[{idx+1}/{genome_number}] genes present: {sample_genes_count}")
            gm = GenomeMinimiser(record=record,
                                 needed_genes_list=needed,
                                 idx=idx,
                                 model_name=model_name)
            seq_id = f"Minimized_E_coli_K12_MG1655_{idx+1}"
            out.write(f">{seq_id}\n{gm.reduced_genome_str}\n")

            genome_length = len(gm.reduced_genome_str)
            sizes_mbp.append(genome_length / 1e6)

            if idx <= 9 or (idx + 1) % 100 == 0:
                red_pct = (original_length - genome_length) / original_length * 100.0
                print(f"  → {genome_length:,} bp ({red_pct:.1f}% reduction)")
                tot_red_pct += red_pct
                total_length_bp += genome_length

    average_reduction_pct = tot_red_pct / genome_number
    average_length_bp = total_length_bp / genome_number

    return {
            "genome_count": genome_number,
            "average_reduction_pct": average_reduction_pct,
            "average_length_bp": average_length_bp,
        }



def process_multiple_genomes_multiple_files(
    genome_path: str,
    genes_path: str,
    model_name: str,
    output_dir: str = None,
    filename_template: str = "minimized_{model}_{idx:04d}.fasta"
):
    """
    Minimize genomes for all lists and save EACH result as its own FASTA file.
    """

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "minimized_genomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load once
    record = SeqIO.read(genome_path, "genbank")
    original_length = len(record.seq)

    all_lists = np.load(genes_path, allow_pickle=True).tolist()
    genome_number = len(all_lists)

    tot_red_pct = 0.0
    total_length = 0

    print(f"Writing {genome_number} individual FASTA files to: {output_dir}")

    for idx, needed in enumerate(all_lists):
        sample_genes_count = len(needed)
        print(f"[{idx+1}/{genome_number}] genes present: {sample_genes_count}")

        gm = GenomeMinimiser(
            record=record,
            needed_genes_list=needed,
            idx=idx,
            model_name=model_name
        )

        seq_id = f"Minimized_E_coli_K12_MG1655_{idx+1}"
        genome_str = gm.reduced_genome_str
        genome_length = len(genome_str)
        red_pct = (original_length - genome_length) / original_length * 100.0

        filename = filename_template.format(model=model_name, idx=idx)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as fh:
            fh.write(f">{seq_id}\n{genome_str}\n")

        tot_red_pct += red_pct
        total_length += genome_length

        if idx <= 9 or (idx + 1) % 100 == 0:
            print(f"  → saved {os.path.basename(out_path)} | {genome_length:,} bp ({red_pct:.1f}% reduction)")

    average_reduction_pct = tot_red_pct / genome_number
    average_length_bp = total_length / genome_number

    return {
        "genome_count": genome_number,
        "average_reduction_pct": average_reduction_pct,
        "average_length_bp": average_length_bp
    }
