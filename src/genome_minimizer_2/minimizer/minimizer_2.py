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
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class GenomeMinimiser:
    '''
    A class used to minimize a genome sequence by removing non-essential genes
    '''
    def __init__(self, record_path: str, needed_genes_path: str, idx: int, model_name: str):
        self.idx = idx
        self.model_name = model_name
        
        # Load genome and genes data
        self.record = self.load_genome(record_path)
        self.wildtype_sequence = self.record  # For consistency with existing code
        self.original_genome_length = len(self.record.seq)
        self.needed_genes = self.get_needed_genes(needed_genes_path)[idx]  # Get genes for this specific index
        
        # Process the genome
        self.features = self._extract_non_essential_genes()
        self.positions_to_remove = self._get_positions_to_remove()
        self.reduced_genome_str = self._create_minimized_sequence()
        self.minimised_genomes_sizes = []
    
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
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
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
            
            if not file_path.endswith((".gb", ".genbank")):
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
            logging.info(f"Attempting to load genes file from {file_path}")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            
            if not file_path.endswith(".npy"):
                raise ValueError(f"The file {file_path} could not be read.\nEnsure the file holds numpy format.")
            
            present_genes = np.load(file_path, allow_pickle=True).tolist()
            logging.info(f"✓ Successfully loaded gene file from {file_path}")
            return present_genes
        
        except FileNotFoundError as fnf_error:
            logging.error(f"\n✗ Error: {fnf_error}")
            raise
        except ValueError as val_error:
            logging.error(f"\n✗ Error: {val_error}")
            raise
        except Exception as e:
            logging.error(f"\n✗ Error: {e}.\nAn unexpected error occurred.")
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
            
            plt.figure(figsize=(8, 6))
            plt.hist(self.minimised_genomes_sizes, bins=10, color="dodgerblue", alpha=0.7)
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
            figures_dir = os.path.join(PROJECT_ROOT, " src/genome_minimizer_2/minimizers/generated_genomes")
            os.makedirs(figures_dir, exist_ok=True)
            
            plt.savefig(
                os.path.join(figures_dir, f"minimised_genomes_distribution_{self.model_name}.pdf"), 
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
    from collections import defaultdict
    
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


def process_multiple_genomes_single_file(genome_path: str, genes_path: str, model_name: str, output_file: str):
    '''
    Process multiple genome minimizations and save to a single FASTA file
    
    Parameters:
    genome_path (str): Path to the GenBank genome file
    genes_path (str): Path to the numpy file containing gene lists
    model_name (str): Name of the model for file naming
    output_file (str): Output FASTA file path
    
    Returns:
    dict: Results including sizes and duplicate statistics
    '''
    if not output_file:
        output_file = "minimized_genomes.fasta"

    try:
        # Load the gene lists
        present_genes = np.load(genes_path, allow_pickle=True).tolist()
        
        minimised_genomes_sizes = []
        sequences_dict = {}
        
        print(f"Starting to create {len(present_genes)} reduced genomes...")
        print(f"Output file: {output_file}")
        
        # Open single output file
        with open(output_file, 'w') as outfile:
            outfile.write(f"Num. Minimized genomes generated using model: {model_name}\n")
            outfile.write(f"Num. Total genomes: {len(present_genes)}\n")
            outfile.write(f"Num. Generated on: {np.datetime64('now')}\n")
            
            for idx in range(len(present_genes)):
                try:
                    minimiser = GenomeMinimiser(genome_path, genes_path, idx, model_name)
                    
                    # Save to the single file
                    sequence_id = f"Minimized_E_coli_K12_MG1655_{idx+1}"
                    outfile.write(f">{sequence_id}\n")
                    outfile.write(f"{minimiser.reduced_genome_str}\n")
                    
                    # Store sequence for duplicate checking
                    sequences_dict[sequence_id] = minimiser.reduced_genome_str
                    
                    # Get statistics
                    stats = minimiser.get_reduction_stats()
                    
                    if (idx + 1) % 100 == 0 or idx < 10:  # Show progress for first 10 and every 100
                        print(f"Processed {idx+1}/{len(present_genes)}: "
                              f"{stats['reduced_length']:,} bp "
                              f"({stats['reduction_percentage']:.1f}% reduction)")
                    
                    minimised_genomes_sizes.append(stats['reduced_length'] / 1e6)
                    
                except Exception as e:
                    logging.error(f"\n✗ Error processing genome {idx+1}: {e}")
                    continue
        
        print(f"\n✓ All genomes saved to: {output_file}")
        
        # Check for duplicates
        print("Analyzing sequence duplicates...")
        duplicate_stats = check_sequence_duplicates(sequences_dict)
        print_duplicate_statistics(duplicate_stats)
        
        # Generate size distribution plot
        if minimised_genomes_sizes and len(minimised_genomes_sizes) >= 10:
            print("Generating size distribution plot...")
            try:
                dummy_minimiser = GenomeMinimiser(genome_path, genes_path, 0, model_name)
                dummy_minimiser.minimised_genomes_sizes = minimised_genomes_sizes
                dummy_minimiser.plot()
                print("✓ Size distribution plot generated")
            except Exception as e:
                logging.warning(f"Could not generate plot: {e}")
        
        # Prepare results
        results = {
            'minimised_genomes_sizes': minimised_genomes_sizes,
            'duplicate_stats': duplicate_stats,
            'output_file': output_file,
            'total_processed': len(minimised_genomes_sizes),
            'statistics': {
                'mean_size_mbp': np.mean(minimised_genomes_sizes),
                'median_size_mbp': np.median(minimised_genomes_sizes),
                'min_size_mbp': np.min(minimised_genomes_sizes),
                'max_size_mbp': np.max(minimised_genomes_sizes),
                'std_size_mbp': np.std(minimised_genomes_sizes)
            }
        }
        
        return results
        
    except Exception as e:
        logging.error(f"\n✗ rror in process_multiple_genomes_single_file: {e}")
        raise


def combine_existing_fasta_files(input_dir: str, output_file: str, check_duplicates: bool = True):
    """
    Combine existing FASTA files into a single file and optionally check for duplicates
    
    Parameters:
    input_dir (str): Directory containing FASTA files
    output_file (str): Output combined FASTA file
    check_duplicates (bool): Whether to check for duplicate sequences
    
    Returns:
    dict: Results including file count and duplicate statistics
    """
    try:
        from Bio import SeqIO
        
        sequences_dict = {}
        file_count = 0
        
        print(f"Combining FASTA files from: {input_dir}")
        
        with open(output_file, 'w') as outfile:
            outfile.write(f"# Combined FASTA sequences from: {input_dir}\n")
            outfile.write(f"# Combined on: {np.datetime64('now')}\n")
            
            for filename in sorted(os.listdir(input_dir)):
                if filename.endswith(('.fasta', '.fa', '.fas')):
                    file_path = os.path.join(input_dir, filename)
                    
                    try:
                        for record in SeqIO.parse(file_path, "fasta"):
                            # Write to combined file
                            outfile.write(f">{record.id}\n")
                            outfile.write(f"{record.seq}\n")
                            
                            # Store for duplicate checking
                            if check_duplicates:
                                sequences_dict[record.id] = str(record.seq)
                        
                        file_count += 1
                        if file_count % 10 == 0:
                            print(f"Processed {file_count} files...")
                            
                    except Exception as e:
                        logging.warning(f"Error processing file {filename}: {e}")
                        continue
        
        print(f"\n✓ Combined {file_count} FASTA files into: {output_file}")
        print(f"- Total sequences: {len(sequences_dict)}")
        
        # Check for duplicates
        duplicate_stats = None
        if check_duplicates and sequences_dict:
            print("Analyzing sequence duplicates...")
            duplicate_stats = check_sequence_duplicates(sequences_dict)
            print_duplicate_statistics(duplicate_stats)
        
        results = {
            'files_processed': file_count,
            'total_sequences': len(sequences_dict),
            'output_file': output_file,
            'duplicate_stats': duplicate_stats
        }
        
        return results
        
    except Exception as e:
        logging.error(f"\n✗ Error combining FASTA files: {e}")
        raise


def reduce_genome_interactive():
    """
    Interactive function to reduce genomes with user input prompts.
    This recreates the original commented-out functionality.
    """
    print("\n" + "="*80)
    print("INTERACTIVE GENOME MINIMIZATION")
    print("="*80)
    
    try:
        # Get user inputs
        genome_path = input("Please enter path for the GenBank sequence file: ").strip()
        genes_path = input("Please enter path to the list of lists of sample genes (.npy): ").strip()
        weight_str = input("Please enter the weight of the model: ").strip()
        
        # Ask for output preference
        output_choice = input("Generate single FASTA file? (y/n, default=y): ").strip().lower()
        single_file = output_choice != 'n'
        
        # Validate inputs
        if not genome_path or not genes_path or not weight_str:
            print("\n✗ All inputs are required")
            return None
        
        try:
            weight = float(weight_str)
        except ValueError:
            print("\n✗ Weight must be a valid number")
            return None
        
        if not os.path.exists(genome_path):
            print(f"\n✗ Genome file not found: {genome_path}")
            return None
            
        if not os.path.exists(genes_path):
            print(f"\n✗ Genes file not found: {genes_path}")
            return None
        
        # Set up output
        base_output_dir = PROJECT_ROOT + "/src/genome_minimizer_2/data/generated_genomes/"
        os.makedirs(base_output_dir, exist_ok=True)
        
        if single_file:
            output_file = os.path.join(base_output_dir, f"minimized_genomes_weight_{weight}.fasta")
            print(f"Will generate single file: {output_file}")
        else:
            output_dir = os.path.join(base_output_dir, f"weight_{weight}/")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Will generate multiple files in: {output_dir}")
        
        # Load basic info for display
        try:
            from Bio import SeqIO
            wildtype_sequence = SeqIO.read(genome_path, "genbank")
            original_genome_length = len(wildtype_sequence.seq)
            present_genes_list = np.load(genes_path, allow_pickle=True).tolist()
        except Exception as e:
            print(f"\n✗ Error loading data: {e}")
            return None
        
        print(f"✓ Loaded genome with {original_genome_length:,} base pairs")
        print(f"✓ Loaded {len(present_genes_list)} gene sets")
        
        # Process genomes
        if single_file:
            results = process_multiple_genomes_single_file(
                genome_path, genes_path, f"weight_{weight}", output_file
            )
            
            print(f"\nFinal Results:")
            stats = results['statistics']
            print(f"- Processed genomes: {results['total_processed']}")
            print(f"- Average size: {stats['mean_size_mbp']:.2f} Mbp")
            print(f"- Size range: {stats['min_size_mbp']:.2f} - {stats['max_size_mbp']:.2f} Mbp")
            print(f"- Compression ratio: {results['duplicate_stats']['compression_ratio']:.1%}")
            
            return results
        else:
            # Use the original multi-file approach
            sizes = reduce_genome_batch(genome_path, genes_path, weight, output_dir)
            return sizes
        
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
        return None
    except Exception as e:
        print(f"\n✗ Error in interactive genome reduction: {e}")
        return None


def reduce_genome_batch(genome_path: str, genes_path: str, weight: float, output_dir: str = None):
    """
    Batch process genome reduction without user interaction.
    
    Parameters:
    genome_path (str): Path to the GenBank genome file
    genes_path (str): Path to the numpy file containing gene lists
    weight (float): Model weight for file naming
    output_dir (str): Output directory (optional)
    
    Returns:
    list: List of minimized genome sizes in Mbp
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT + "data/generated_genomes/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        present_genes_list = np.load(genes_path, allow_pickle=True).tolist()
        
        print(f"Processing {len(present_genes_list)} genomes...")
        minimised_genomes_sizes = []
        
        for idx, _ in enumerate(present_genes_list):
            minimiser = GenomeMinimiser(genome_path, genes_path, idx, f"weight_{weight}")
            
            minimized_genome_filename = os.path.join(
                output_dir, 
                f"minimized_genome_{minimiser.idx}.fasta"
            )
            
            minimiser.save_minimized_genome(minimized_genome_filename)
            minimised_genomes_sizes.append(len(minimiser.reduced_genome_str) / 1e6)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(present_genes_list)} genomes...")
        
        # Generate plot if enough data
        if len(minimised_genomes_sizes) >= 10:
            temp_minimiser = GenomeMinimiser(genome_path, genes_path, 0, f"weight_{weight}")
            temp_minimiser.minimised_genomes_sizes = minimised_genomes_sizes
            temp_minimiser.plot()
        
        return minimised_genomes_sizes
        
    except Exception as e:
        logging.error(f"\n✗ Error in batch genome reduction: {e}")
        raise