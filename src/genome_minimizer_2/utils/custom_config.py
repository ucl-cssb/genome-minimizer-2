#!/usr/bin/env python3
"""
Configs for VAE trianign.
"""

# import libraries
from dataclasses import dataclass, fields
from typing import Dict, Any
import json
import argparse
from pathlib import Path

@dataclass
class ExperimentConfig:
    """
    Configuration for experiments
    """
    
    # Model parameters
    hidden_dim: int = 512
    latent_dim: int = 32
    
    # Training parameters  
    n_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_norm: float = 1.0
    lambda_l1: float = 0.01
    
    # Loss scheduling parameters
    min_beta: float = 0.0
    max_beta: float = 1.0
    gamma_start: float = 1.0
    gamma_end: float = 0.1
    weight: float = 1.0  # For v3
    
    # Trainer version
    trainer_version: str = "v2"  # v0, v1, v2, v3
    
    # Scheduler parameters
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    
    # Data split parameters
    test_size: float = 0.3
    val_ratio: float = 0.3333
    random_state: int = 12345
    
    # Output parameters
    experiment_name: str = "experiment"
    save_model: bool = True
    generate_plots: bool = True
    calculate_metrics: bool = True
    explore_latent_space: bool = True
    
    def interactive_override(self):
        """Allow user to interactively override parameters"""
        print("\n" + "="*60)
        print("INTERACTIVE PARAMETER OVERRIDE")
        print("="*60)
        print("Press Enter to keep default value, or type new value to override.")
        print("Type 'skip' to skip all remaining parameters.")
        print("-" * 60)
        
        for field_info in fields(self):
            current_value = getattr(self, field_info.name)
            field_type = field_info.type
            
            # Handle special types
            if field_type == bool:
                prompt = f"{field_info.name} [{current_value}] (true/false): "
            elif field_type == str and field_info.name == "trainer_version":
                prompt = f"{field_info.name} [{current_value}] (v0/v1/v2/v3): "
            else:
                prompt = f"{field_info.name} [{current_value}]: "
            
            try:
                user_input = input(prompt).strip()
                
                if user_input.lower() == 'skip':
                    print("Skipping remaining parameters...")
                    break
                    
                if user_input == '':
                    continue
                    
                # Convert input to appropriate type
                if field_type == bool:
                    new_value = user_input.lower() in ['true', 't', '1', 'yes', 'y']
                elif field_type == int:
                    new_value = int(user_input)
                elif field_type == float:
                    new_value = float(user_input)
                elif field_type == str:
                    new_value = user_input
                else:
                    new_value = user_input
                
                setattr(self, field_info.name, new_value)
                print(f"✓ Updated {field_info.name} to {new_value}")
                
            except ValueError as e:
                print(f"✗ Invalid input for {field_info.name}: {e}")
                print(f"  Keeping default value: {current_value}")
            except KeyboardInterrupt:
                print("\n\n✗ Process interrupted by user")
                break
    
    def update_from_dict(self, overrides: Dict[str, Any]):
        """Update configuration from dictionary"""
        updated_params = []
        invalid_params = []
        
        for key, value in overrides.items():
            if hasattr(self, key):
                try:
                    # Get the field type for validation
                    field_type = next(f.type for f in fields(self) if f.name == key)
                    
                    # Convert value to appropriate type if needed
                    if field_type == bool and isinstance(value, str):
                        value = value.lower() in ['true', 't', '1', 'yes', 'y']
                    elif field_type in [int, float] and isinstance(value, str):
                        value = field_type(value)
                    
                    setattr(self, key, value)
                    updated_params.append(f"{key}: {value}")
                    
                except (ValueError, TypeError) as e:
                    invalid_params.append(f"{key}: {e}")
            else:
                invalid_params.append(f"{key}: parameter not found")
        
        if updated_params:
            print("\n✓ Updated parameters:")
            for param in updated_params:
                print(f"  {param}")
        
        if invalid_params:
            print("\n✗ Invalid parameters:")
            for param in invalid_params:
                print(f"  {param}")
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {field.name: getattr(self, field.name) for field in fields(self)}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    def load_from_json(self, filepath: str):
        """Load configuration from JSON file"""
        if not Path(filepath).exists():
            print(f"Configuration file {filepath} not found.")
            return
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.update_from_dict(config_dict)
        print(f"Configuration loaded from {filepath}")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create config from command line arguments"""
        config = cls()
        
        # Convert args to dict and filter out None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        # Remove non-config arguments
        non_config_args = {'mode', 'config_file', 'interactive'}
        for arg in non_config_args:
            args_dict.pop(arg, None)
        
        if args_dict:
            config.update_from_dict(args_dict)
        
        return config


def add_config_arguments(parser: argparse.ArgumentParser):
    """Add all configuration parameters as command line arguments"""
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--hidden-dim', type=int, help='Hidden dimension size')
    model_group.add_argument('--latent-dim', type=int, help='Latent dimension size')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--n-epochs', type=int, help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, help='Batch size')
    train_group.add_argument('--learning-rate', type=float, help='Learning rate')
    train_group.add_argument('--max-norm', type=float, help='Max gradient norm')
    train_group.add_argument('--lambda-l1', type=float, help='L1 regularization weight')
    
    # Loss scheduling parameters
    loss_group = parser.add_argument_group('Loss Scheduling Parameters')
    loss_group.add_argument('--min-beta', type=float, help='Minimum beta value')
    loss_group.add_argument('--max-beta', type=float, help='Maximum beta value')
    loss_group.add_argument('--gamma-start', type=float, help='Starting gamma value')
    loss_group.add_argument('--gamma-end', type=float, help='Ending gamma value')
    loss_group.add_argument('--weight', type=float, help='Weight parameter for v3')
    
    # Trainer version
    trainer_group = parser.add_argument_group('Trainer Parameters')
    trainer_group.add_argument('--trainer-version', choices=['v0', 'v1', 'v2', 'v3'], 
                              help='Trainer version')
    
    # Scheduler parameters
    sched_group = parser.add_argument_group('Scheduler Parameters')
    sched_group.add_argument('--scheduler-step-size', type=int, help='Scheduler step size')
    sched_group.add_argument('--scheduler-gamma', type=float, help='Scheduler gamma')
    
    # Data split parameters
    data_group = parser.add_argument_group('Data Split Parameters')
    data_group.add_argument('--test-size', type=float, help='Test split size')
    data_group.add_argument('--val-ratio', type=float, help='Validation ratio')
    data_group.add_argument('--random-state', type=int, help='Random state seed')
    
    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--experiment-name', type=str, help='Experiment name')
    output_group.add_argument('--save-model', action='store_true', help='Save model')
    output_group.add_argument('--no-save-model', action='store_false', dest='save_model', 
                             help='Do not save model')
    output_group.add_argument('--generate-plots', action='store_true', help='Generate plots')
    output_group.add_argument('--no-generate-plots', action='store_false', dest='generate_plots',
                             help='Do not generate plots')
    output_group.add_argument('--calculate-metrics', action='store_true', help='Calculate metrics')
    output_group.add_argument('--no-calculate-metrics', action='store_false', dest='calculate_metrics',
                             help='Do not calculate metrics')
    output_group.add_argument('--explore-latent-space', action='store_true', help='Explore latent space')
    output_group.add_argument('--no-explore-latent-space', action='store_false', dest='explore_latent_space',
                             help='Do not explore latent space')
    
    # Configuration file and interactive mode
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config-file', type=str, help='Load configuration from JSON file')
    config_group.add_argument('--interactive', action='store_true', 
                             help='Interactive parameter override mode')


# Example usage functions
def setup_experiment_config(args):
    """Setup experiment configuration with various override options"""
    
    # Start with default config
    config = ExperimentConfig()
    
    # Load from file if specified
    if hasattr(args, 'config_file') and args.config_file:
        config.load_from_json(args.config_file)
    
    # Override with command line arguments
    if args:
        config = ExperimentConfig.from_args(args)
    
    # Interactive mode
    if hasattr(args, 'interactive') and args.interactive:
        config.interactive_override()
    
    return config


def create_sample_config_file():
    """Create a sample configuration file for users"""
    config = ExperimentConfig()
    config.save_to_json('sample_config.json')
    print("Sample configuration file created: sample_config.json")
    print("You can modify this file and load it with --config-file sample_config.json")


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Experiment Configuration Example')
    parser.add_argument('--mode', choices=['experiment', 'other'], default='experiment')
    
    # Add all configuration arguments
    add_config_arguments(parser)
    
    args = parser.parse_args()
    
    if args.mode == 'experiment':
        print("\n" + "="*80)
        print("NEW EXPERIMENT RUN")
        print("="*80)
        
        config = setup_experiment_config(args)