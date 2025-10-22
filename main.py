#!/usr/bin/env python3
"""
MAIN CLI SCRIPT FOR COMPLETE GENOMICS VAE PIPELINE
"""

# Import libraries
import os
import sys
import torch
import argparse
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent))

# Import directories first
from src.genome_minimizer_2.utils.directories import (
    TEN_K_DATASET_FULL,
    TEN_K_DATASET_PHYLOGROUPS_FULL,
    PAPER_ESSENTIAL_GENES_FULL,
    ESSENTIAL_GENES_POSITIONS,
    WILD_TYPE_SEQUENCE_FULL,
    SEQUENCES_FULL,
    SEQUENCE_OUT
)
import src.genome_minimizer_2.explore_data.data_exploration
import src.genome_minimizer_2.explore_data.extract_essential_genes
from src.genome_minimizer_2.utils.extras import write_samples_to_dataframe
from src.genome_minimizer_2.explore_data.data_exploration import load_and_validate_data
from src.genome_minimizer_2.explore_data.binary_converter import masks_to_gene_lists

# Set run device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_banner():
    """GENOME MINIMIZER 2 banner"""
    banner = """
            ┌──────────────────────────────────────────────────────────────────────┐
            │  ██████╗ ███████╗███╗   ██╗ ██████╗ ███╗   ███╗███████╗    ██████╗   │
            │ ██╔════╝ ██╔════╝████╗  ██║██╔═══██╗████╗ ████║██╔════╝    ╚════██╗  │
            │ ██║  ███╗█████╗  ██╔██╗ ██║██║   ██║██╔████╔██║█████╗       █████╔╝  │
            │ ██║   ██║██╔══╝  ██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══╝      ██╔═══╝   │
            │ ╚██████╔╝███████╗██║ ╚████║╚██████╔╝██║ ╚═╝ ██║███████╗    ███████╗  │
            │  ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝    ╚══════╝  │
            │                                                                      │
            │ ███╗   ███╗██╗███╗   ██╗██╗███╗   ███╗██╗███████╗███████╗██████╗     │
            │ ████╗ ████║██║████╗  ██║██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗    │
            │ ██╔████╔██║██║██╔██╗ ██║██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝    │
            │ ██║╚██╔╝██║██║██║╚██╗██║██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗    │
            │ ██║ ╚═╝ ██║██║██║ ╚████║██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║    │
            │ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝    │
            └──────────────────────────────────────────────────────────────────────┘
            """
    print(banner)



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run integrated VAE genomics experiments')
    
    # Base argiments
    parser.add_argument('--mode', 
                        choices=['training', 'experiment', 'minimizer', 'explore', 'preprocess', 'sample', 'convert-samples'],
                        default='training', 
                        help='Run mode: training experiment, custom experiment, evaluate existing model, genome minimizer, data exploration, preprocessing, or sampling')
    
    # Training/experiment arguments
    parser.add_argument('--preset', 
                        choices=['v0', 'v1', 'v2', 'v3'], 
                        default='v3',
                        help='Which model preset to run (for training mode)')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=None,
                        help='Override number of epochs')
    
    parser.add_argument('--model-path', 
                        type=str,
                        help='Path to existing model file (.pt) for sampling mode')
    
    # Sampling arguments
    parser.add_argument('--genome-path',
                        type=str,
                        default=WILD_TYPE_SEQUENCE_FULL,
                        help='Path to GenBank genome file (.gb or .genbank)')
    
    parser.add_argument('--genes-path',
                        type=str,
                        help='Path to numpy file containing gene lists (.npy)')
    
    parser.add_argument('--output-dir',
                        type=str,
                        default='./minimized_genomes',
                        help='Output directory for minimized genomes (multiple files)')
    
    parser.add_argument('--output-file',
                        type=str,
                        help='Output file path for single combined FASTA file')
    
    parser.add_argument('--single-file',
                        action='store_true',
                        help='Generate single FASTA file instead of multiple files')
    
    parser.add_argument('--model-name',
                        type=str,
                        default='default',
                        help='Model name for file naming (genome minimizer)')
    
    parser.add_argument('--num-samples',
                        type=int,
                        default=1,
                        help='Number of samples to generate')
    
    parser.add_argument('--sampling-mode',
                        choices=['default', 'focused'],
                        default='default',
                        help='Sampling mode: default (random) or focused (around specific latent point)')
    
    parser.add_argument('--noise-level',
                        type=float,
                        default=0.1,
                        help='Noise level for focused sampling')
    
    # Data preprocessing arguments
    parser.add_argument('--force-reprocess',
                        action='store_true',
                        help='Force reprocessing of essential gene positions even if file exists')
    
    # Parse known args first to check the mode
    known_args, _ = parser.parse_known_args()
    
    # Add experiment-specific arguments only if in experiment mode
    if known_args.mode == 'experiment':
        try:
            from src.genome_minimizer_2.utils.custom_config import add_config_arguments
            add_config_arguments(parser)
        except ImportError:
            print("✗  Could not load experiment config - experiment mode may not work")
    
    return parser.parse_args()


def check_data_availability():
    """Check if required data files are available"""
    required_files = {
        'Main Dataset': TEN_K_DATASET_FULL,
        'Phylogroups': TEN_K_DATASET_PHYLOGROUPS_FULL,
        'Essential Genes': PAPER_ESSENTIAL_GENES_FULL
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("✗  Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("- Please ensure all data files are in the correct locations.")
        return False
    
    print("✓ All required data files found")
    return True


def run_data_exploration():
    """Run data exploration analysis"""
    print("\n" + "="*80)
    print("DATA EXPLORATION AND ANALYSIS")
    print("="*80)
    
    try:
        # Import and run the polished data exploration
        src.genome_minimizer_2.explore_data.data_exploration.main()
        print("✓ Data exploration completed successfully")
        print("- Check the data_exploration/figures/ directory for generated plots")
        return True
        
    except Exception as e:
        print(f"✗ Error during data exploration: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_preprocessing(force_reprocess=False):
    """Run data preprocessing using the polished script"""
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Check if essential gene positions already exist
    if os.path.exists(ESSENTIAL_GENES_POSITIONS) and not force_reprocess:
        print(f"✓ Essential gene positions already exist: {ESSENTIAL_GENES_POSITIONS}")
        print("Use --force-reprocess to regenerate\n")
        return True
    
    try:
        # Import and run the polished preprocessing
        src.genome_minimizer_2.explore_data.extract_essential_genes.main()
        print("✓ Essential gene positions generated successfully")
        print(f"Saved to: {ESSENTIAL_GENES_POSITIONS}\n")
        return True
        
    except Exception as e:
        print(f"✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_sampling(args):
    """Run model sampling analysis"""
    print("\n" + "="*80)
    print("MODEL SAMPLING")
    print("="*80)
    
    # Validate model file
    if not args.model_path:
        print("✗ Model path required for sampling mode")
        print("Use --model-path to specify the trained model file\n")
        return False
    
    if not os.path.exists(args.model_path):
        print(f"✗ Model file not found: {args.model_path}")
        return False
    
    if not os.path.exists(args.genes_path):
        print(f"✗ Model file not found: {args.genes_path}. Run preprocessing first.")
        return False
    
    try:
        # Import sampling utilities
        from src.genome_minimizer_2.utils.extras import (
            sample_from_model, 
            get_latent_variables, 
            count_essential_genes,
            plot_samples_distribution,
            plot_essential_genes_distribution,
            plot_essential_vs_total,
            load_model
        )
        import pandas as pd
        import numpy as np
        import pickle
        from sklearn.decomposition import PCA
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        
        # Load and prepare data
        print("Loading dataset...")
        _, merged_df, _ = load_and_validate_data()

        data_array_t, phylogroups_array = merged_df.iloc[:, :-1].values, merged_df.iloc[:, -1].values
        all_genes = merged_df.columns[:-1]
        
        # Create data loaders
        data_tensor = torch.tensor(data_array_t, dtype=torch.float32)
        _, temp_data, _, temp_labels = train_test_split(
            data_tensor, phylogroups_array, test_size=0.3, random_state=12345
        )
        _, test_data, _, test_labels = train_test_split(
            temp_data, temp_labels, test_size=0.3333, random_state=12345
        )

        with open(args.genes_path, 'rb') as f:
            essential_gene_positions = pickle.load(f)
        # essential_gene_positions = np.load(args.genes_path, allow_pickle=True)
        
        test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)
        
        # Determine model parameters from data
        input_dim = data_array_t.shape[1]
        print(f"Detected input dimension: {input_dim}")
        
        # Try to load model and infer architecture
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration if available
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']

        else:
            model_filename = Path(args.model_path).name.lower()
            if 'v0' in model_filename:
                model_name = 'v0'
            elif 'v1' in model_filename:
                model_name = 'v1'
            elif 'v2' in model_filename:
                model_name = 'v2'
            elif 'v3' in model_filename:
                model_name = 'v3'
            else:
                print("✗ Could not detect version")
                return False

            # Then handle config loading
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
            else:
                from src.genome_minimizer_2.utils.experiments import (
                    get_v0_config, get_v1_config, get_v2_config, get_v3_config
                )
                if model_name == 'v0':
                    config = get_v0_config()
                elif model_name == 'v1':
                    config = get_v1_config()
                elif model_name == 'v2':
                    config = get_v2_config()
                elif model_name == 'v3':
                    config = get_v3_config()
        
        try:
            # print(f"PROJECT_ROOT: {PROJECT_ROOT}")
            output_dir = Path(PROJECT_ROOT) / "models" / f"{model_name}_model" / "sampling_results"
            output_dir.mkdir(parents=True, exist_ok=True)            
            print(f"✓ Created output_dir: {output_dir}")
            
        except Exception as e:
            print(f"✗ Error creating directories: {e}")
            return False

        # Load the model
        model = load_model(input_dim, config.hidden_dim, config.latent_dim, args.model_path)

        print(f"\n{'='*80}")
        print(f"Sampling Configuration:")
        print(f"- Model: {Path(args.model_path).name}")
        print(f"- Architecture: {input_dim} -> {config.hidden_dim} -> {config.latent_dim}")
        print(f"- Samples: {args.num_samples}")
        print(f"- Mode: {args.sampling_mode}")
        print(f"- Output: {output_dir}")
        print(f"{'='*80}")
        
        if args.sampling_mode == 'default':
            # Default sampling
            print("Generating default samples...")
            binary_samples, continuous_samples, z = sample_from_model(
                model, config.latent_dim, args.num_samples, device
            )
            
        else:
            # Focused sampling
            print("Generating focused samples...")
            # First generate a small batch to find interesting latent point
            binary_temp, continuous_temp, z_temp = sample_from_model(
                model, config.latent_dim, 100, device
            )
            
            # Find the sample with minimum genes (most minimal genome)
            min_ones_index = np.argmin(binary_temp.sum(axis=1))
            latent_distances = np.linalg.norm(continuous_temp - continuous_temp[min_ones_index], axis=1)
            closest_latent_index = np.argmin(latent_distances)
            
            # Generate samples around this point
            z_of_interest = z_temp[closest_latent_index].unsqueeze(0)
            with torch.no_grad():
                noise = torch.randn(args.num_samples, config.latent_dim, device=device) * args.noise_level
                continuous_samples = model.decode(z_of_interest + noise).cpu().numpy()
            
            binary_samples = (continuous_samples > 0.5).astype(float)
            z = z_of_interest + noise
        
        # Analyze results
        genome_sizes = binary_samples.sum(axis=1)
        essential_counts = count_essential_genes(binary_samples, essential_gene_positions)
        
        print(f"\n✓ Sampling Results:")
        print(f"- Generated samples: {binary_samples.shape[0]}")
        print(f"- Median genome size: {np.median(genome_sizes):.0f} genes")
        print(f"- Genome size range: {np.min(genome_sizes):.0f} - {np.max(genome_sizes):.0f}")
        print(f"- Median essential genes: {np.median(essential_counts):.0f}")
        print(f"- Essential range: {np.min(essential_counts):.0f} - {np.max(essential_counts):.0f}")
        
        # Generate plots
        print("\nGenerating analysis plots...")
        
        # Genome size distribution
        plot_samples_distribution(
            binary_samples,
            str(output_dir / f"genome_size_distribution_{args.sampling_mode}.pdf"),
            "dodgerblue",
            int(np.min(genome_sizes) * 0.9),
            int(np.max(genome_sizes) * 1.1)
        )
        
        # Essential genes distribution
        plot_essential_genes_distribution(
            essential_counts,
            str(output_dir / f"essential_genes_distribution_{args.sampling_mode}.pdf"),
            "violet",
            int(np.min(essential_counts) * 0.9),
            int(np.max(essential_counts) * 1.1)
        )
        
        # Essential vs total relationship
        plot_essential_vs_total(
            essential_counts,
            genome_sizes,
            str(output_dir / f"essential_vs_total_{args.sampling_mode}.pdf")
        )
        
        # Latent space analysis
        print("Analyzing latent space...")
        latents = get_latent_variables(model, test_loader, device)
        data_pca = PCA(n_components=2).fit_transform(latents)
        df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
        df_pca['phylogroup'] = test_labels
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='phylogroup', data=df_pca)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Latent Space PCA by Phylogroup')
        plt.tight_layout()
        plt.savefig(output_dir / f"latent_space_pca_{args.sampling_mode}.pdf", 
                   format="pdf", bbox_inches="tight")
        plt.close()
        
        # Save data
        print("Saving results...")
        np.save(output_dir / f"binary_samples_{args.sampling_mode}.npy", binary_samples)
        write_samples_to_dataframe(binary_samples, all_genes, f"{output_dir}/data_full_samples_df.csv")
        
        print(f"\n✓ SAMPLING COMPLETE!")
        print(f"- Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_single_experiment(args):
    """Run a training experiment with preset configuration"""
    print("\n" + "="*80)
    print("TRAINING EXPERIMENT RUN")
    print("="*80)

    try:
        from src.genome_minimizer_2.utils.experiments import (
            IntegratedExperimentRunner, 
            get_v0_config, get_v1_config, get_v2_config, get_v3_config
        )
        
        # Get configuration based on preset
        if args.preset == 'v0':
            config = get_v0_config()
        elif args.preset == 'v1':
            config = get_v1_config()
        elif args.preset == 'v2':
            config = get_v2_config()
        elif args.preset == 'v3':
            config = get_v3_config()
        
        # Override epochs if specified
        if args.epochs:
            config.n_epochs = args.epochs

        print(f"\n{'='*80}")
        print(f"Running {config.experiment_name} experiment")
        print(f"Hidden dim: {config.hidden_dim}, Latent dim: {config.latent_dim}")
        print(f"Epochs: {config.n_epochs}, Trainer: {config.trainer_version}")
        print(f"{'='*80}")
        
        runner = IntegratedExperimentRunner(config)
        results = runner.run_complete_experiment()
        
        print(f"\n{config.experiment_name.upper()} COMPLETED!")
        if 'f1_overall' in results:
            print(f"F1 Score: {results['f1_overall']:.3f}")
            print(f"Accuracy: {results['accuracy_overall']:.3f}")
        
        return results
        
    except ImportError as e:
        print(f"✗ Could not import experiment modules: {e}")
        return None


def run_custom_experiment(args):
    """Run experiment with custom configuration"""
    print("\n" + "="*80)
    print("CUSTOM EXPERIMENT RUN")
    print("="*80)

    try:
        from src.genome_minimizer_2.utils.custom_config import setup_experiment_config
        from src.genome_minimizer_2.utils.experiments import IntegratedExperimentRunner
        
        config = setup_experiment_config(args)

        print(f"\n{'='*80}")
        print(f"Running {config.experiment_name} experiment")
        print(f"Hidden dim: {config.hidden_dim}, Latent dim: {config.latent_dim}")
        print(f"Epochs: {config.n_epochs}, Trainer: {config.trainer_version}")
        print(f"{'='*80}")

        runner = IntegratedExperimentRunner(config)
        results = runner.run_complete_experiment()

        print(f"\n{config.experiment_name.upper()} COMPLETED!")
        if 'f1_overall' in results:
            print(f"F1 Score: {results['f1_overall']:.3f}")
            print(f"Accuracy: {results['accuracy_overall']:.3f}")
        
        return results
        
    except ImportError as e:
        print(f"✗ Could not import experiment modules: {e}")
        return None

def run_genome_minimizer(args):
    """Run the genome minimizer"""
    print("\n" + "="*80)
    print("GENOME MINIMIZER RUN")
    print("="*80)
    
    # Validate required files
    if not os.path.exists(args.genome_path):
        print(f"✗ Genome file not found: {args.genome_path}")
        return None
    
    if not args.genes_path:
        print("✗ Genes path required for genome minimizer")
        return None
        
    if not os.path.exists(args.genes_path):
        print(f"✗ Genes file not found: {args.genes_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Processing genome: {Path(args.genome_path).name}")
    print(f"Using genes from: {Path(args.genes_path).name}")
    print(f"Model name: {args.model_name}")
    print(f"{'='*80}")
    
    try:
        from src.genome_minimizer_2.minimizer.minimizer_2 import (
            process_multiple_genomes_single_file, 
        )
        
        # Create unified output directory first
        if args.output_file:
            # If specific output file given, use its directory
            output_dir = Path(args.output_file).parent
            output_filename = Path(args.output_file).name
        elif args.single_file:
            # Single file mode with default name
            output_dir = Path(args.output_dir)
            output_filename = f"minimized_genomes_{args.model_name}.fasta"
        else:
            # Multiple files mode
            output_dir = Path(args.output_dir)
            output_filename = None
        
        # Create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        
        # Determine final output path
        if args.single_file or args.output_file:
            output_file = output_dir / output_filename
            print(f"Generating single FASTA file: {output_file}")
            
            results = process_multiple_genomes_single_file(
                genome_path=args.genome_path,
                genes_path=args.genes_path,
                model_name=args.model_name,
                output_file=str(output_file)  # Convert Path to string
            )
            
            print("\n✓ GENOME MINIMIZATION COMPLETED!")
            print(f"- Single file generated: {output_file}")
            print(f"- Processed: {results['total_processed']} genomes")
            print(f"- Unique sequences: {results['duplicate_stats']['unique_sequences']}")
            print(f"- Compression ratio: {results['duplicate_stats']['compression_ratio']:.1%}")
            
            return results
            
        else:
            print(f"Generating multiple files in: {output_dir}")
            
            minimised_sizes = process_multiple_genomes_single_file(
                genome_path=args.genome_path,
                genes_path=args.genes_path,
                model_name=args.model_name,
                output_dir=str(output_dir)  # Pass the directory
            )
            
            print("\n✓ GENOME MINIMIZATION COMPLETED!")
            print(f"- Files saved to: {output_dir}")
            print(f"- Processed {len(minimised_sizes)} genome variants")
            if minimised_sizes:
                avg_size = sum(minimised_sizes) / len(minimised_sizes)
                print(f"- Average size: {avg_size:.2f} Mbp")
                print(f"- Size range: {min(minimised_sizes):.2f} - {max(minimised_sizes):.2f} Mbp")
            
            return minimised_sizes

    except Exception as e:
        print(f"✗ Error during genome minimization: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    args = parse_arguments()
    
    print_banner()
    print(f"\nRunning in {args.mode} mode on {device}")
    
    if args.mode in ['training', 'experiment', 'explore', 'preprocess', 'sample']:
        if not check_data_availability():
            print("\n✗ Cannot proceed without required data files")
            return 1
    
    results = None
    
    try:
        if args.mode == 'explore':
            success = run_data_exploration()
            return 0 if success else 1
            
        elif args.mode == 'preprocess':
            success = run_preprocessing(args.force_reprocess)
            return 0 if success else 1
            
        elif args.mode == 'sample':
            success = run_sampling(args)
            return 0 if success else 1
            
        elif args.mode == 'training':
            results = run_single_experiment(args)
            
        elif args.mode == 'experiment':
            results = run_custom_experiment(args)
            
        elif args.mode == 'minimizer':
            results = run_genome_minimizer(args)

        elif args.mode == 'convert-samples':
            dataset_csv = TEN_K_DATASET_FULL
            masks_npy = SEQUENCES_FULL
            out_ids_npy = SEQUENCE_OUT

            large_data = pd.read_csv(dataset_csv, index_col=0)
            data_without_lineage = large_data.drop(index=["Lineage"], errors="ignore")
            print(f"Dataset shape (samples x genes): {data_without_lineage.shape}")
            data_transpose = data_without_lineage.transpose()
            cols = data_transpose.columns

            masks_to_gene_lists(
                masks_npy_path=masks_npy,
                cols=cols,
                out_ids_npy=out_ids_npy,
            )
    
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED!")
    print("="*80)
    
    if args.mode in ['training', 'experiment']:
        print("- Check the models/ directory for results and plots\n")
    elif args.mode == 'minimizer':
        if args.single_file or args.output_file:
            print("- Check for the generated FASTA file\n")
        else:
            print(f"- Check the {args.output_dir}/ directory for minimized genomes\n")
    elif args.mode == 'sample':
        print("- Check the sampling_results/ directory for sampling analysis\n")
    elif args.mode == 'explore':
        print("- Check the data_exploration/figures/ directory for analysis plots\n")
    elif args.mode == 'preprocess':
        print("- Essential gene positions saved to data/ directory\n")
    elif args.mode == 'convert-samples':
        print("- Binary sample data converted to gene names\n")
    
    
    return 0 if results is not None else 1

if __name__ == "__main__":
    exit(main())