"""
Graph Construction Module for Music Recommendation GNN

This module converts preprocessed data into PyTorch Geometric heterogeneous graphs
ready for GNN training.
"""

import torch
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import ToUndirected
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MusicGraphConstructor:
    """
    Constructs PyTorch Geometric graphs from preprocessed music data
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # Data containers
        self.mappings = None
        self.entity_counts = None
        self.edges = None
        self.features = None
        self.splits = None
        self.metadata = None

        # Graph objects
        self.full_graph = None
        self.train_graph = None
        self.val_graph = None
        self.test_graph = None

        print(f"📂 Initializing graph constructor with data from: {data_dir}")

    def load_preprocessed_data(self):
        """Load all preprocessed components"""
        print("📥 LOADING PREPROCESSED DATA")
        print("=" * 40)

        try:
            # Load mappings
            with open(f"{self.data_dir}/mappings.pkl", 'rb') as f:
                self.mappings = pickle.load(f)
            print(f"✅ Loaded entity mappings")

            # Load entity counts
            with open(f"{self.data_dir}/entity_counts.pkl", 'rb') as f:
                self.entity_counts = pickle.load(f)
            print(f"✅ Loaded entity counts: {sum(self.entity_counts.values()):,} total nodes")

            # Load edges
            edges_data = np.load(f"{self.data_dir}/edges.npz")
            self.edges = {key: edges_data[key] for key in edges_data.keys()}
            total_edges = sum(len(edges) for edges in self.edges.values())
            print(f"✅ Loaded edge data: {total_edges:,} total edges")

            # Load features
            features_data = np.load(f"{self.data_dir}/features.npz")
            self.features = {key: features_data[key] for key in features_data.keys()}
            print(f"✅ Loaded node features for {len(self.features)} entity types")

            # Load splits
            splits_data = np.load(f"{self.data_dir}/splits.npz", allow_pickle=True)
            self.splits = {key: splits_data[key] for key in splits_data.keys()}
            print(f"✅ Loaded train/val/test splits")

            # Load metadata
            with open(f"{self.data_dir}/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata")

            print()
            self._print_data_summary()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def _print_data_summary(self):
        """Print comprehensive summary of loaded data"""
        print("📊 DETAILED DATA SUMMARY")
        print("=" * 40)

        # Entity counts
        print("🎵 Node Types:")
        for entity_type, count in self.entity_counts.items():
            print(f"   • {entity_type.capitalize()}: {count:,} nodes")

        print(f"\n🔗 Edge Types:")
        for edge_type, edges in self.edges.items():
            print(f"   • {edge_type}: {len(edges):,} edges")

        print(f"\n🎨 Feature Dimensions:")
        for entity_type, features in self.features.items():
            print(f"   • {entity_type.capitalize()}: {features.shape}")

        print(f"\n✂️  Data Splits:")
        print(f"   • Training edges: {len(self.splits['train_edges']):,}")
        print(f"   • Validation edges: {len(self.splits['val_edges']):,}")
        print(f"   • Test edges: {len(self.splits['test_edges']):,}")
        print()

    def create_node_mappings(self) -> Dict[str, torch.Tensor]:
        """Create global node ID mappings for the heterogeneous graph"""
        print("🗺️  CREATING GLOBAL NODE MAPPINGS")
        print("=" * 40)

        # Create continuous node IDs across all entity types
        global_node_mapping = {}
        reverse_global_mapping = {}
        current_offset = 0

        # Use the same order as in your entity_counts
        entity_order = list(self.entity_counts.keys())

        for entity_type in entity_order:
            count = self.entity_counts[entity_type]

            # Create global node IDs
            global_ids = torch.arange(current_offset, current_offset + count)
            global_node_mapping[entity_type] = global_ids

            # Create reverse mapping for debugging
            reverse_global_mapping[entity_type] = (current_offset, current_offset + count)

            print(f"   • {entity_type.capitalize()}: {current_offset:,} → {current_offset + count - 1:,}")
            current_offset += count

        print(f"\n📈 Total global nodes: {current_offset:,}")
        print()

        self.global_node_mapping = global_node_mapping
        self.reverse_global_mapping = reverse_global_mapping

        return global_node_mapping

    def create_hetero_graph(self, edge_subset: Optional[Dict[str, np.ndarray]] = None) -> HeteroData:
        """Create a heterogeneous graph using PyTorch Geometric"""
        print("🏗️  CREATING HETEROGENEOUS GRAPH")
        print("=" * 40)

        # Use full edges if no subset provided
        if edge_subset is None:
            edge_subset = self.edges

        # Initialize heterogeneous graph
        graph = HeteroData()

        # Add node features for each entity type
        print("🎨 Adding node features...")

        # Create mapping between entity_counts keys and features keys
        feature_key_mapping = {}

        # Check for exact matches first
        for entity_type in self.entity_counts.keys():
            if entity_type in self.features:
                feature_key_mapping[entity_type] = entity_type
            else:
                # Check for singular/plural variations
                possible_keys = [
                    entity_type.rstrip('s'),  # Remove 's' if present
                    entity_type + 's',  # Add 's' if not present
                    entity_type.replace('s', ''),  # Remove all 's'
                ]

                for possible_key in possible_keys:
                    if possible_key in self.features:
                        feature_key_mapping[entity_type] = possible_key
                        break

        print(f"   📝 Feature key mapping: {feature_key_mapping}")

        # Map entity count keys to graph node type names (singular)
        entity_to_node_type = {}
        for entity_type in self.entity_counts.keys():
            # Convert plural entity names to singular for graph
            if entity_type.endswith('s'):
                node_type = entity_type.rstrip('s')
            else:
                node_type = entity_type
            entity_to_node_type[entity_type] = node_type

        for entity_type in self.entity_counts.keys():
            if entity_type in feature_key_mapping:
                feature_key = feature_key_mapping[entity_type]
                node_type = entity_to_node_type[entity_type]

                # Convert to torch tensor and add to graph
                node_features = torch.tensor(self.features[feature_key], dtype=torch.float32)
                graph[node_type].x = node_features

                print(
                    f"   • {entity_type.capitalize()}: {node_features.shape} (key: '{feature_key}' -> node: '{node_type}')")
            else:
                node_type = entity_to_node_type[entity_type]
                print(f"   ⚠️  {entity_type.capitalize()}: No features found for node type '{node_type}'")
                print(f"      Available feature keys: {list(self.features.keys())}")

        # Add edges for each relationship type
        print(f"\n🔗 Adding edge connections...")

        # Define edge type mappings for heterogeneous graph (using singular node types)
        edge_type_mappings = {
            'playlist_track': ('playlist', 'contains', 'track'),
            'track_artist': ('track', 'performed_by', 'artist'),
            'track_album': ('track', 'belongs_to', 'album'),
            'user_playlist': ('user', 'created', 'playlist'),
            'playlist_user': ('playlist', 'created_by', 'user')
        }

        for edge_type, edges in edge_subset.items():
            if edge_type in edge_type_mappings and len(edges) > 0:
                src_type, relation, dst_type = edge_type_mappings[edge_type]

                # Convert edges to torch tensor and transpose for PyG format
                edge_index = torch.tensor(edges.T, dtype=torch.long)

                # Add to graph
                graph[src_type, relation, dst_type].edge_index = edge_index

                print(f"   • {src_type} --{relation}--> {dst_type}: {len(edges):,} edges")

        print(f"\n✅ Heterogeneous graph created successfully")
        print(f"   📊 Total node types: {len(graph.node_types)}")
        print(f"   🔗 Total edge types: {len(graph.edge_types)}")

        # Verify nodes have features
        print(f"\n🔍 Node feature verification:")
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                print(f"   ✅ {node_type}: {graph[node_type].x.shape}")
            else:
                print(f"   ❌ {node_type}: Missing features")

        return graph

    def create_train_val_test_graphs(self):
        """Create separate graphs for training, validation, and testing"""
        print("✂️  CREATING TRAIN/VALIDATION/TEST GRAPHS")
        print("=" * 50)

        # Create edge subsets for each split
        train_edges = {
            'playlist_track': self.splits['train_edges'],
            'track_artist': self.edges['track_artist'],  # Keep all structural edges
            'track_album': self.edges['track_album'],
            'user_playlist': self.edges['user_playlist'],
            'playlist_user': self.edges['playlist_user']
        }

        val_edges = {
            'playlist_track': self.splits['val_edges'],
            'track_artist': self.edges['track_artist'],
            'track_album': self.edges['track_album'],
            'user_playlist': self.edges['user_playlist'],
            'playlist_user': self.edges['playlist_user']
        }

        test_edges = {
            'playlist_track': self.splits['test_edges'],
            'track_artist': self.edges['track_artist'],
            'track_album': self.edges['track_album'],
            'user_playlist': self.edges['user_playlist'],
            'playlist_user': self.edges['playlist_user']
        }

        # Create graphs
        print("🚂 Creating training graph...")
        self.train_graph = self.create_hetero_graph(train_edges)

        print("\n🔍 Creating validation graph...")
        self.val_graph = self.create_hetero_graph(val_edges)

        print("\n🧪 Creating test graph...")
        self.test_graph = self.create_hetero_graph(test_edges)

        print(f"\n📊 GRAPH SPLIT SUMMARY:")
        print(f"   🚂 Training playlist-track edges: {len(train_edges['playlist_track']):,}")
        print(f"   🔍 Validation playlist-track edges: {len(val_edges['playlist_track']):,}")
        print(f"   🧪 Test playlist-track edges: {len(test_edges['playlist_track']):,}")
        print()

    def create_full_graph(self):
        """Create the complete graph with all edges"""
        print("🌐 CREATING COMPLETE GRAPH")
        print("=" * 40)

        self.full_graph = self.create_hetero_graph()
        print()

    def analyze_graph_structure(self):
        """Analyze and visualize graph properties"""
        print("🔍 GRAPH STRUCTURE ANALYSIS")
        print("=" * 40)

        if self.full_graph is None:
            print("❌ No graph created yet. Run create_full_graph() first.")
            return

        # Node degree analysis
        print("📊 Node Degree Analysis:")

        for edge_type in self.full_graph.edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = self.full_graph[edge_type].edge_index

            if len(edge_index) > 0:
                # Source node degrees (out-degree)
                src_degrees = torch.bincount(edge_index[0])
                dst_degrees = torch.bincount(edge_index[1])

                print(f"\n   🔗 {src_type} --{relation}--> {dst_type}:")
                print(f"      • Avg out-degree ({src_type}): {src_degrees.float().mean():.2f}")
                print(f"      • Max out-degree ({src_type}): {src_degrees.max().item()}")
                print(f"      • Avg in-degree ({dst_type}): {dst_degrees.float().mean():.2f}")
                print(f"      • Max in-degree ({dst_type}): {dst_degrees.max().item()}")

        # Graph connectivity
        print(f"\n🌐 Graph Connectivity:")
        total_nodes = sum(self.entity_counts.values())
        total_edges = sum(len(self.full_graph[et].edge_index[0]) for et in self.full_graph.edge_types)

        print(f"   • Total nodes: {total_nodes:,}")
        print(f"   • Total edges: {total_edges:,}")
        print(f"   • Average degree: {(2 * total_edges / total_nodes):.2f}")

        # Memory usage estimation
        node_memory = sum(self.features[et].nbytes for et in self.features) / 1024 ** 2
        edge_memory = sum(edges.nbytes for edges in self.edges.values()) / 1024 ** 2

        print(f"\n💾 Memory Usage:")
        print(f"   • Node features: {node_memory:.1f} MB")
        print(f"   • Edge indices: {edge_memory:.1f} MB")
        print(f"   • Total estimated: {node_memory + edge_memory:.1f} MB")
        print()

    def save_graphs(self, output_dir: str):
        """Save constructed graphs to disk"""
        print("💾 SAVING GRAPH OBJECTS")
        print("=" * 40)

        os.makedirs(output_dir, exist_ok=True)

        # Save individual graphs
        if self.full_graph is not None:
            torch.save(self.full_graph, f"{output_dir}/full_graph.pt")
            print(f"✅ Saved full graph to {output_dir}/full_graph.pt")

        if self.train_graph is not None:
            torch.save(self.train_graph, f"{output_dir}/train_graph.pt")
            print(f"✅ Saved training graph to {output_dir}/train_graph.pt")

        if self.val_graph is not None:
            torch.save(self.val_graph, f"{output_dir}/val_graph.pt")
            print(f"✅ Saved validation graph to {output_dir}/val_graph.pt")

        if self.test_graph is not None:
            torch.save(self.test_graph, f"{output_dir}/test_graph.pt")
            print(f"✅ Saved test graph to {output_dir}/test_graph.pt")

        # Save graph metadata
        graph_metadata = {
            'created_at': datetime.now().isoformat(),
            'total_nodes': sum(self.entity_counts.values()),
            'total_edges': sum(len(edges) for edges in self.edges.values()),
            'node_types': list(self.entity_counts.keys()),
            'edge_types': list(self.edges.keys()),
            'entity_counts': self.entity_counts,
            'feature_shapes': {k: list(v.shape) for k, v in self.features.items()},
            'split_sizes': {
                'train': len(self.splits['train_edges']),
                'val': len(self.splits['val_edges']),
                'test': len(self.splits['test_edges'])
            }
        }

        with open(f"{output_dir}/graph_metadata.json", 'w') as f:
            json.dump(graph_metadata, f, indent=2)

        print(f"✅ Saved graph metadata to {output_dir}/graph_metadata.json")
        print(f"\n🎉 All graph objects saved to: {output_dir}")
        print()

    def verify_graphs(self):
        """Verify that all graphs are constructed correctly"""
        print("✅ GRAPH VERIFICATION")
        print("=" * 40)

        graphs_to_check = [
            ('Full Graph', self.full_graph),
            ('Train Graph', self.train_graph),
            ('Val Graph', self.val_graph),
            ('Test Graph', self.test_graph)
        ]

        for graph_name, graph in graphs_to_check:
            if graph is not None:
                print(f"\n🔍 {graph_name}:")
                print(f"   • Node types: {len(graph.node_types)}")
                print(f"   • Edge types: {len(graph.edge_types)}")

                # Check for NaN values in features
                for node_type in graph.node_types:
                    if hasattr(graph[node_type], 'x'):
                        features = graph[node_type].x
                        nan_count = torch.isnan(features).sum().item()
                        print(f"   • {node_type} features: {features.shape}, NaN count: {nan_count}")

                # Check edge indices
                total_edges = 0
                for edge_type in graph.edge_types:
                    edge_index = graph[edge_type].edge_index
                    total_edges += edge_index.shape[1]

                    # Check for invalid indices
                    src_type, relation, dst_type = edge_type

                    # Create a mapping from graph node types to entity count keys
                    node_type_to_count_key = {}
                    for count_key in self.entity_counts.keys():
                        # Handle plural to singular mapping
                        if count_key.endswith('s'):
                            singular = count_key.rstrip('s')
                            node_type_to_count_key[singular] = count_key
                        node_type_to_count_key[count_key] = count_key

                    # Get entity counts using the mapping
                    src_count_key = node_type_to_count_key.get(src_type)
                    dst_count_key = node_type_to_count_key.get(dst_type)

                    if src_count_key and dst_count_key:
                        max_src = self.entity_counts[src_count_key] - 1
                        max_dst = self.entity_counts[dst_count_key] - 1

                        invalid_src = (edge_index[0] > max_src).sum().item()
                        invalid_dst = (edge_index[1] > max_dst).sum().item()

                        if invalid_src > 0 or invalid_dst > 0:
                            print(f"   ⚠️  {src_type}--{relation}-->{dst_type}: Invalid indices detected!")
                            print(f"      Invalid src: {invalid_src}, Invalid dst: {invalid_dst}")
                        else:
                            print(f"   ✅ {src_type}--{relation}-->{dst_type}: All indices valid")
                    else:
                        print(f"   ⚠️  {src_type}--{relation}-->{dst_type}: Could not map to entity counts")
                        print(f"      Available entity count keys: {list(self.entity_counts.keys())}")

                print(f"   • Total edges: {total_edges:,}")
            else:
                print(f"\n❌ {graph_name}: Not created")

        print(f"\n🎯 Verification complete!")
        print()

    def run_complete_pipeline(self, output_dir: str = None) -> Dict[str, HeteroData]:
        """
        Run the complete graph construction pipeline

        Args:
            output_dir: Directory to save graphs (optional)

        Returns:
            Dictionary containing all constructed graphs
        """
        print("🚀 STARTING COMPLETE GRAPH CONSTRUCTION PIPELINE")
        print("=" * 60)

        # Step 1: Load preprocessed data
        self.load_preprocessed_data()

        # Step 2: Create node mappings
        global_mappings = self.create_node_mappings()

        # Step 3: Create the complete graph
        self.create_full_graph()

        # Step 4: Create train/validation/test graphs
        self.create_train_val_test_graphs()

        # Step 5: Analyze graph structure
        self.analyze_graph_structure()

        # Step 6: Verify all graphs
        self.verify_graphs()

        # Step 7: Save graphs to disk (if output_dir provided)
        if output_dir:
            self.save_graphs(output_dir)

        print("🎉 GRAPH CONSTRUCTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ All graph objects are ready for GNN training")

        # Return the graphs for immediate use
        return {
            'full_graph': self.full_graph,
            'train_graph': self.train_graph,
            'val_graph': self.val_graph,
            'test_graph': self.test_graph
        }


def load_graph(graph_path: str) -> HeteroData:
    """
    Load a saved graph from disk

    Args:
        graph_path: Path to the saved graph file

    Returns:
        Loaded HeteroData graph
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    return torch.load(graph_path)


def load_all_graphs(graphs_dir: str) -> Dict[str, HeteroData]:
    """
    Load all saved graphs from a directory

    Args:
        graphs_dir: Directory containing saved graph files

    Returns:
        Dictionary containing all loaded graphs
    """
    graph_files = {
        'full_graph': f"{graphs_dir}/full_graph.pt",
        'train_graph': f"{graphs_dir}/train_graph.pt",
        'val_graph': f"{graphs_dir}/val_graph.pt",
        'test_graph': f"{graphs_dir}/test_graph.pt"
    }

    graphs = {}
    for graph_name, graph_path in graph_files.items():
        if os.path.exists(graph_path):
            graphs[graph_name] = load_graph(graph_path)
            print(f"✅ Loaded {graph_name}")
        else:
            print(f"⚠️  {graph_name} not found at {graph_path}")

    return graphs


if __name__ == "__main__":
    # Example usage
    data_directory = "../data/processed/gnn_ready"
    output_directory = "../data/processed/graphs"

    # Initialize and run graph construction
    constructor = MusicGraphConstructor(data_directory)
    graphs = constructor.run_complete_pipeline(output_directory)

    print(f"\n📊 GRAPH ACCESS:")
    for name, graph in graphs.items():
        if graph is not None:
            print(f"   • {name}: {len(graph.node_types)} node types, {len(graph.edge_types)} edge types")