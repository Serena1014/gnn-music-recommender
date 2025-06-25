#!/usr/bin/env python3
"""
Data Loader Utility for GNN Music Recommender

This module provides easy-to-use functions for loading preprocessed data
and preparing it for GNN training and evaluation.

Usage:
    from src.utils.data_loader import load_preprocessed_data, get_data_splits

    # Load all data
    data = load_preprocessed_data("data/processed/gnn_ready")

    # Get train/val/test splits
    train_data, val_data, test_data = get_data_splits(data)
"""

import pickle
import numpy as np
import json
import os
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GNNDataLoader:
    """Utility class for loading and managing preprocessed GNN data"""

    def __init__(self, data_dir: str):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing preprocessed data files
        """
        self.data_dir = data_dir
        self.data = {}
        self._validate_data_directory()

    def _validate_data_directory(self):
        """Validate that all required files exist"""
        required_files = [
            'mappings.pkl',
            'reverse_mappings.pkl',
            'entity_counts.pkl',
            'edges.npz',
            'features.npz',
            'splits.npz',
            'metadata.json'
        ]

        missing_files = []
        for file_name in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file_name)):
                missing_files.append(file_name)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {self.data_dir}: {missing_files}"
            )

    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all preprocessed data

        Returns:
            Dictionary containing all loaded data
        """
        logger.info(f"🔄 Loading preprocessed data from: {self.data_dir}")

        # Load mappings
        with open(os.path.join(self.data_dir, 'mappings.pkl'), 'rb') as f:
            self.data['mappings'] = pickle.load(f)

        with open(os.path.join(self.data_dir, 'reverse_mappings.pkl'), 'rb') as f:
            self.data['reverse_mappings'] = pickle.load(f)

        # Load entity counts
        with open(os.path.join(self.data_dir, 'entity_counts.pkl'), 'rb') as f:
            self.data['entity_counts'] = pickle.load(f)

        # Load edges
        edges_data = np.load(os.path.join(self.data_dir, 'edges.npz'), allow_pickle=True)
        self.data['edges'] = {key: edges_data[key] for key in edges_data.keys()}

        # Load features
        features_data = np.load(os.path.join(self.data_dir, 'features.npz'), allow_pickle=True)
        self.data['features'] = {key: features_data[key] for key in features_data.keys()}

        # Load splits
        splits_data = np.load(os.path.join(self.data_dir, 'splits.npz'), allow_pickle=True)
        self.data['splits'] = {}
        for key in splits_data.keys():
            value = splits_data[key]
            # Handle different data types properly
            if key == 'split_ratios' and value.dtype == object:
                # Extract dict from object array
                self.data['splits'][key] = value.item() if value.size == 1 else value
            else:
                self.data['splits'][key] = value

        # Load metadata
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            self.data['metadata'] = json.load(f)

        logger.info("✅ All data loaded successfully")
        return self.data

    def get_entity_info(self) -> Dict[str, int]:
        """Get information about entities in the graph"""
        if 'entity_counts' not in self.data:
            self.load_all_data()

        return self.data['entity_counts']

    def get_graph_edges(self, edge_type: str = None) -> Dict[str, np.ndarray]:
        """
        Get graph edges

        Args:
            edge_type: Specific edge type to return, or None for all edges

        Returns:
            Dictionary of edges or specific edge array
        """
        if 'edges' not in self.data:
            self.load_all_data()

        if edge_type:
            return self.data['edges'].get(edge_type)
        return self.data['edges']

    def get_node_features(self, node_type: str = None) -> Dict[str, np.ndarray]:
        """
        Get node features

        Args:
            node_type: Specific node type to return, or None for all features

        Returns:
            Dictionary of features or specific feature array
        """
        if 'features' not in self.data:
            self.load_all_data()

        if node_type:
            return self.data['features'].get(node_type)
        return self.data['features']

    def get_data_splits(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get train/validation/test splits

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if 'splits' not in self.data:
            self.load_all_data()

        splits = self.data['splits']

        train_data = {
            'positive_edges': splits['train_edges'],
            'indices': splits['train_indices']
        }

        val_data = {
            'positive_edges': splits['val_edges'],
            'negative_edges': splits['negative_val'],
            'indices': splits['val_indices']
        }

        test_data = {
            'positive_edges': splits['test_edges'],
            'negative_edges': splits['negative_test'],
            'indices': splits['test_indices']
        }

        return train_data, val_data, test_data

    def get_metadata(self) -> Dict[str, Any]:
        """Get preprocessing metadata"""
        if 'metadata' not in self.data:
            self.load_all_data()

        return self.data['metadata']

    def print_data_summary(self):
        """Print a summary of the loaded data"""
        if not self.data:
            self.load_all_data()

        print("📊 GNN DATA SUMMARY")
        print("=" * 40)

        # Entity counts
        entity_counts = self.data['entity_counts']
        print("🏷️  Entities:")
        for entity_type, count in entity_counts.items():
            print(f"   • {entity_type}: {count:,}")

        # Edge counts
        print("\n🔗 Edges:")
        for edge_type, edges in self.data['edges'].items():
            print(f"   • {edge_type}: {len(edges):,}")

        # Feature dimensions
        print("\n🎨 Features:")
        for node_type, features in self.data['features'].items():
            print(f"   • {node_type}: {features.shape}")

        # Split sizes
        splits = self.data['splits']
        print("\n📊 Data Splits:")
        print(f"   • Train: {len(splits['train_edges']):,} edges")
        print(f"   • Validation: {len(splits['val_edges']):,} edges")
        print(f"   • Test: {len(splits['test_edges']):,} edges")

        # Metadata
        metadata = self.data['metadata']
        print(f"\n⚙️  Configuration:")
        print(f"   • Created: {metadata['created_at']}")
        print(f"   • Split strategy: {metadata['preprocessing_notes']['split_strategy']}")
        print(f"   • Random seed: {metadata['preprocessing_notes'].get('random_seed', 'Not specified')}")


# Convenience functions for easy use
def load_preprocessed_data(data_dir: str) -> Dict[str, Any]:
    """
    Load all preprocessed data from directory

    Args:
        data_dir: Directory containing preprocessed data files

    Returns:
        Dictionary containing all loaded data
    """
    loader = GNNDataLoader(data_dir)
    return loader.load_all_data()


def get_data_splits(data_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Get train/validation/test splits

    Args:
        data_dir: Directory containing preprocessed data files

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    loader = GNNDataLoader(data_dir)
    return loader.get_data_splits()


def print_data_summary(data_dir: str):
    """
    Print summary of preprocessed data

    Args:
        data_dir: Directory containing preprocessed data files
    """
    loader = GNNDataLoader(data_dir)
    loader.print_data_summary()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Explore preprocessed GNN data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing preprocessed data')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only print summary, don\'t load all data')

    args = parser.parse_args()

    try:
        if args.summary_only:
            print_data_summary(args.data_dir)
        else:
            # Load and explore data
            loader = GNNDataLoader(args.data_dir)
            data = loader.load_all_data()
            loader.print_data_summary()

            # Example: Access specific data
            train_data, val_data, test_data = loader.get_data_splits()
            print(f"\n🔍 Example Access:")
            print(f"   • Train edges shape: {train_data['positive_edges'].shape}")
            print(f"   • Validation positive edges: {len(val_data['positive_edges'])}")
            print(f"   • Validation negative edges: {len(val_data['negative_edges'])}")

    except Exception as e:
        print(f"❌ Error: {e}")