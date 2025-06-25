#!/usr/bin/env python3
"""
Runner script for data preprocessing with YAML configuration support

This script provides an easy way to run the data preprocessing pipeline
using either command line arguments or a YAML configuration file.

Usage:
    # Using command line arguments
    python run_preprocessing.py --input_file data/input.json --output_dir data/output

    # Using configuration file
    python run_preprocessing.py --config config/preprocessing_config.yaml

    # Override config with command line
    python run_preprocessing.py --config config/preprocessing_config.yaml --output_dir custom/output
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add src directory to path to import our preprocessing module
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data_preprocessing import main as preprocessing_main
except ImportError:
    print("❌ Could not import data_preprocessing module.")
    print("   Make sure the data_preprocessing.py file is in the src/ directory")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration: {e}")
        sys.exit(1)


def merge_args_with_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """Merge command line arguments with configuration file"""

    # Command line arguments take precedence over config file
    if not args.input_file and 'data' in config:
        args.input_file = config['data'].get('input_file')

    if not args.output_dir and 'data' in config:
        args.output_dir = config['data'].get('output_dir')

    # Split ratios
    if 'splits' in config:
        if args.train_ratio == 0.7:  # Default value, use config
            args.train_ratio = config['splits'].get('train_ratio', 0.7)
        if args.val_ratio == 0.15:  # Default value, use config
            args.val_ratio = config['splits'].get('val_ratio', 0.15)
        if args.test_ratio == 0.15:  # Default value, use config
            args.test_ratio = config['splits'].get('test_ratio', 0.15)
        if args.random_seed == 42:  # Default value, use config
            args.random_seed = config['splits'].get('random_seed', 42)

    return args


def validate_args(args: argparse.Namespace):
    """Validate arguments"""
    if not args.input_file:
        print("❌ Error: input_file is required")
        sys.exit(1)

    if not args.output_dir:
        print("❌ Error: output_dir is required")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file does not exist: {args.input_file}")
        sys.exit(1)

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Split ratios must sum to 1.0, got {total_ratio}")
        sys.exit(1)

    print("✅ Arguments validated successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Run data preprocessing for GNN music recommender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with command line arguments
    python run_preprocessing.py --input_file data/sampled.json --output_dir data/gnn_ready

    # Using configuration file
    python run_preprocessing.py --config config/preprocessing_config.yaml

    # Override config file settings
    python run_preprocessing.py --config config/preprocessing_config.yaml --output_dir custom_output
        """
    )

    # Configuration file option
    parser.add_argument('--config', type=str,
                        help='Path to YAML configuration file')

    # Core arguments
    parser.add_argument('--input_file', type=str,
                        help='Path to input sampled JSON file')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for preprocessed data')

    # Split configuration
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        args = merge_args_with_config(args, config)

    # Validate arguments
    validate_args(args)

    # Print configuration
    print("\n🔧 PREPROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"📂 Input file: {args.input_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Train ratio: {args.train_ratio}")
    print(f"📊 Val ratio: {args.val_ratio}")
    print(f"📊 Test ratio: {args.test_ratio}")
    print(f"🎲 Random seed: {args.random_seed}")
    if args.config:
        print(f"⚙️  Config file: {args.config}")
    print()

    # Create a mock sys.argv for the preprocessing script
    original_argv = sys.argv
    sys.argv = [
        'data_preprocessing.py',
        '--input_file', args.input_file,
        '--output_dir', args.output_dir,
        '--train_ratio', str(args.train_ratio),
        '--val_ratio', str(args.val_ratio),
        '--test_ratio', str(args.test_ratio),
        '--random_seed', str(args.random_seed)
    ]

    try:
        # Run the preprocessing
        preprocessing_main()
        print("\n🎉 Preprocessing completed successfully!")
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()