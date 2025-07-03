"""
Graph utilities for music recommendation GNN project.

This module provides utility functions for working with heterogeneous graphs,
including data loading, graph analysis, and common operations.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from pathlib import Path


class GraphUtils:
    """Utility class for graph operations"""

    @staticmethod
    def load_graph(graph_path: str) -> HeteroData:
        """Load a single graph from disk"""
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        return torch.load(graph_path)

    @staticmethod
    def load_graphs(graphs_dir: str, graph_names: Optional[List[str]] = None) -> Dict[str, HeteroData]:
        """
        Load multiple graphs from a directory

        Args:
            graphs_dir: Directory containing graph files
            graph_names: List of specific graph names to load (optional)

        Returns:
            Dictionary of loaded graphs
        """
        if graph_names is None:
            graph_names = ['full_graph', 'train_graph', 'val_graph', 'test_graph']

        graphs = {}
        for name in graph_names:
            graph_path = os.path.join(graphs_dir, f"{name}.pt")
            if os.path.exists(graph_path):
                graphs[name] = GraphUtils.load_graph(graph_path)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️  {name} not found at {graph_path}")

        return graphs

    @staticmethod
    def get_graph_info(graph: HeteroData) -> Dict:
        """Get detailed information about a graph"""
        info = {
            'node_types': list(graph.node_types),
            'edge_types': list(graph.edge_types),
            'node_counts': {},
            'edge_counts': {},
            'feature_dims': {},
            'total_nodes': 0,
            'total_edges': 0
        }

        # Node information
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                num_nodes = graph[node_type].x.shape[0]
                feature_dim = graph[node_type].x.shape[1]
                info['node_counts'][node_type] = num_nodes
                info['feature_dims'][node_type] = feature_dim
                info['total_nodes'] += num_nodes

        # Edge information
        for edge_type in graph.edge_types:
            if hasattr(graph[edge_type], 'edge_index'):
                num_edges = graph[edge_type].edge_index.shape[1]
                info['edge_counts'][str(edge_type)] = num_edges
                info['total_edges'] += num_edges

        return info

    @staticmethod
    def print_graph_summary(graph: HeteroData, graph_name: str = "Graph"):
        """Print a formatted summary of graph statistics"""
        info = GraphUtils.get_graph_info(graph)

        print(f"\n📊 {graph_name.upper()} SUMMARY")
        print("=" * 40)

        print("🎵 Node Types:")
        for node_type, count in info['node_counts'].items():
            feature_dim = info['feature_dims'].get(node_type, 0)
            print(f"   • {node_type}: {count:,} nodes, {feature_dim} features")

        print("\n🔗 Edge Types:")
        for edge_type, count in info['edge_counts'].items():
            print(f"   • {edge_type}: {count:,} edges")

        print(f"\n📈 Totals:")
        print(f"   • Total nodes: {info['total_nodes']:,}")
        print(f"   • Total edges: {info['total_edges']:,}")
        print()

    @staticmethod
    def extract_edge_data(graph: HeteroData, edge_type: Tuple[str, str, str]) -> torch.Tensor:
        """
        Extract edge indices for a specific edge type

        Args:
            graph: HeteroData graph
            edge_type: Tuple of (src_type, relation, dst_type)

        Returns:
            Edge index tensor
        """
        if edge_type in graph.edge_types:
            return graph[edge_type].edge_index
        else:
            raise ValueError(f"Edge type {edge_type} not found in graph")

    @staticmethod
    def get_positive_edges(graph: HeteroData, edge_type: Tuple[str, str, str]) -> torch.Tensor:
        """
        Get positive edges for link prediction as [num_edges, 2] tensor

        Args:
            graph: HeteroData graph
            edge_type: Tuple of (src_type, relation, dst_type)

        Returns:
            Positive edges as [num_edges, 2] tensor
        """
        edge_index = GraphUtils.extract_edge_data(graph, edge_type)
        return edge_index.T  # Transpose to get [num_edges, 2]

    @staticmethod
    def sample_negative_edges(graph: HeteroData,
                              edge_type: Tuple[str, str, str],
                              num_neg_samples: int,
                              existing_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample negative edges for link prediction

        Args:
            graph: HeteroData graph
            edge_type: Tuple of (src_type, relation, dst_type)
            num_neg_samples: Number of negative samples to generate
            existing_edges: Existing positive edges to avoid (optional)

        Returns:
            Negative edges as [num_neg_samples, 2] tensor
        """
        src_type, _, dst_type = edge_type

        # Get number of nodes for each type
        num_src_nodes = graph[src_type].x.shape[0]
        num_dst_nodes = graph[dst_type].x.shape[0]

        # Get existing edges if not provided
        if existing_edges is None:
            existing_edges = GraphUtils.get_positive_edges(graph, edge_type)

        # Create set of existing edges for fast lookup
        existing_set = set()
        for edge in existing_edges:
            existing_set.add((edge[0].item(), edge[1].item()))

        # Sample negative edges
        negative_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 10  # Prevent infinite loop

        while len(negative_edges) < num_neg_samples and attempts < max_attempts:
            src_idx = torch.randint(0, num_src_nodes, (1,)).item()
            dst_idx = torch.randint(0, num_dst_nodes, (1,)).item()

            if (src_idx, dst_idx) not in existing_set:
                negative_edges.append([src_idx, dst_idx])
                existing_set.add((src_idx, dst_idx))

            attempts += 1

        if len(negative_edges) < num_neg_samples:
            print(f"⚠️  Could only sample {len(negative_edges)} negative edges out of {num_neg_samples} requested")

        return torch.tensor(negative_edges, dtype=torch.long)

    @staticmethod
    def get_node_features(graph: HeteroData, node_type: str) -> torch.Tensor:
        """Get node features for a specific node type"""
        if node_type in graph.node_types and hasattr(graph[node_type], 'x'):
            return graph[node_type].x
        else:
            raise ValueError(f"Node type {node_type} not found or has no features")

    @staticmethod
    def check_graph_consistency(graph: HeteroData) -> Dict[str, bool]:
        """
        Check graph consistency and return validation results

        Returns:
            Dictionary with validation results
        """
        results = {
            'has_node_features': True,
            'has_edges': True,
            'valid_edge_indices': True,
            'no_nan_features': True,
            'edge_index_format': True
        }

        # Check node features
        for node_type in graph.node_types:
            if not hasattr(graph[node_type], 'x') or graph[node_type].x is None:
                results['has_node_features'] = False
                print(f"❌ Missing features for node type: {node_type}")
            else:
                # Check for NaN values
                if torch.isnan(graph[node_type].x).any():
                    results['no_nan_features'] = False
                    print(f"❌ NaN values found in features for node type: {node_type}")

        # Check edges
        if len(graph.edge_types) == 0:
            results['has_edges'] = False
            print("❌ Graph has no edges")

        # Check edge indices
        for edge_type in graph.edge_types:
            if hasattr(graph[edge_type], 'edge_index'):
                edge_index = graph[edge_type].edge_index

                # Check format [2, num_edges]
                if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                    results['edge_index_format'] = False
                    print(f"❌ Invalid edge index format for {edge_type}: {edge_index.shape}")

                # Check for valid indices
                src_type, _, dst_type = edge_type
                if hasattr(graph[src_type], 'x') and hasattr(graph[dst_type], 'x'):
                    max_src = graph[src_type].x.shape[0] - 1
                    max_dst = graph[dst_type].x.shape[0] - 1

                    if (edge_index[0] > max_src).any() or (edge_index[1] > max_dst).any():
                        results['valid_edge_indices'] = False
                        print(f"❌ Invalid edge indices for {edge_type}")

        return results

    @staticmethod
    def create_subgraph(graph: HeteroData,
                        node_subset: Dict[str, torch.Tensor]) -> HeteroData:
        """
        Create a subgraph with specified nodes

        Args:
            graph: Original HeteroData graph
            node_subset: Dictionary mapping node types to node indices to keep

        Returns:
            New HeteroData graph with subsetted nodes and edges
        """
        subgraph = HeteroData()

        # Create node mappings (old_idx -> new_idx)
        node_mappings = {}

        # Add nodes and features
        for node_type, node_indices in node_subset.items():
            if node_type in graph.node_types:
                # Subset node features
                if hasattr(graph[node_type], 'x'):
                    subgraph[node_type].x = graph[node_type].x[node_indices]

                # Create mapping from old indices to new indices
                node_mappings[node_type] = {
                    old_idx.item(): new_idx
                    for new_idx, old_idx in enumerate(node_indices)
                }

        # Add edges (filter and remap indices)
        for edge_type in graph.edge_types:
            src_type, relation, dst_type = edge_type

            if src_type in node_subset and dst_type in node_subset:
                edge_index = graph[edge_type].edge_index
                src_mapping = node_mappings[src_type]
                dst_mapping = node_mappings[dst_type]

                # Filter edges that exist in both node subsets
                valid_edges = []
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    dst_idx = edge_index[1, i].item()

                    if src_idx in src_mapping and dst_idx in dst_mapping:
                        new_src = src_mapping[src_idx]
                        new_dst = dst_mapping[dst_idx]
                        valid_edges.append([new_src, new_dst])

                if valid_edges:
                    subgraph[edge_type].edge_index = torch.tensor(
                        valid_edges, dtype=torch.long
                    ).T

        return subgraph


def load_graph_metadata(graphs_dir: str) -> Dict:
    """Load graph metadata from JSON file"""
    metadata_path = os.path.join(graphs_dir, "graph_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠️  Metadata file not found: {metadata_path}")
        return {}


def prepare_link_prediction_data(graph: HeteroData,
                                 edge_type: Tuple[str, str, str],
                                 neg_sampling_ratio: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Prepare data for link prediction task

    Args:
        graph: HeteroData graph
        edge_type: Edge type to predict
        neg_sampling_ratio: Ratio of negative to positive samples

    Returns:
        Dictionary containing positive and negative edges
    """
    # Get positive edges
    pos_edges = GraphUtils.get_positive_edges(graph, edge_type)
    num_pos = pos_edges.shape[0]

    # Sample negative edges
    num_neg = int(num_pos * neg_sampling_ratio)
    neg_edges = GraphUtils.sample_negative_edges(graph, edge_type, num_neg, pos_edges)

    return {
        'pos_edges': pos_edges,
        'neg_edges': neg_edges,
        'edge_type': edge_type
    }


def get_graph_statistics(graphs_dir: str) -> Dict:
    """Get statistics for all graphs in a directory"""
    try:
        graphs = GraphUtils.load_graphs(graphs_dir)
        stats = {}

        for name, graph in graphs.items():
            stats[name] = GraphUtils.get_graph_info(graph)

        return stats
    except Exception as e:
        print(f"Error loading graph statistics: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    graphs_dir = "../data/processed/graphs"

    # Load and analyze graphs
    try:
        graphs = GraphUtils.load_graphs(graphs_dir)

        for name, graph in graphs.items():
            GraphUtils.print_graph_summary(graph, name)

            # Check consistency
            consistency = GraphUtils.check_graph_consistency(graph)
            if all(consistency.values()):
                print(f"✅ {name} passed all consistency checks")
            else:
                print(f"⚠️  {name} has consistency issues")

    except Exception as e:
        print(f"Error: {e}")