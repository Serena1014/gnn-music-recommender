#!/usr/bin/env python3
"""
Test script for graph construction functionality.

This script runs various tests to ensure the graph construction pipeline works correctly.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pickle
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

# Check for PyTorch Geometric
try:
    import torch
    import torch_geometric
    from torch_geometric.data import HeteroData

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch Geometric not found. Please install it:")
    print("   pip install torch-geometric torch-scatter torch-sparse")
    TORCH_GEOMETRIC_AVAILABLE = False

# Only import if torch geometric is available
if TORCH_GEOMETRIC_AVAILABLE:
    from graph_construction import MusicGraphConstructor
    from graph_utils import GraphUtils, prepare_link_prediction_data


def create_mock_data(data_dir: str):
    """Create mock data for testing when real data is not available"""
    os.makedirs(data_dir, exist_ok=True)

    print(f"📝 Creating mock data in {data_dir}")

    # Mock mappings
    mappings = {
        'playlists': {f'playlist_{i}': i for i in range(100)},
        'tracks': {f'track_{i}': i for i in range(500)},
        'artists': {f'artist_{i}': i for i in range(200)},
        'albums': {f'album_{i}': i for i in range(150)},
        'users': {f'user_{i}': i for i in range(80)}
    }

    with open(f"{data_dir}/mappings.pkl", 'wb') as f:
        pickle.dump(mappings, f)

    # Mock entity counts
    entity_counts = {
        'playlists': 100,
        'tracks': 500,
        'artists': 200,
        'albums': 150,
        'users': 80
    }

    with open(f"{data_dir}/entity_counts.pkl", 'wb') as f:
        pickle.dump(entity_counts, f)

    # Mock edges
    edges = {
        'playlist_track': np.random.randint(0, [100, 500], (1000, 2)),
        'track_artist': np.random.randint(0, [500, 200], (500, 2)),
        'track_album': np.random.randint(0, [500, 150], (500, 2)),
        'user_playlist': np.random.randint(0, [80, 100], (120, 2)),
        'playlist_user': np.random.randint(0, [100, 80], (120, 2))
    }

    np.savez(f"{data_dir}/edges.npz", **edges)

    # Mock features
    features = {
        'playlists': np.random.randn(100, 8).astype(np.float32),
        'tracks': np.random.randn(500, 16).astype(np.float32),
        'artists': np.random.randn(200, 12).astype(np.float32),
        'albums': np.random.randn(150, 10).astype(np.float32),
        'users': np.random.randn(80, 6).astype(np.float32)
    }

    np.savez(f"{data_dir}/features.npz", **features)

    # Mock splits
    train_size = int(0.7 * 1000)
    val_size = int(0.15 * 1000)

    splits = {
        'train_edges': edges['playlist_track'][:train_size],
        'val_edges': edges['playlist_track'][train_size:train_size + val_size],
        'test_edges': edges['playlist_track'][train_size + val_size:]
    }

    np.savez(f"{data_dir}/splits.npz", **splits)

    # Mock metadata
    metadata = {
        'created_at': '2024-01-01T00:00:00',
        'total_entities': sum(entity_counts.values()),
        'total_edges': sum(len(e) for e in edges.values())
    }

    with open(f"{data_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f)

    print("✅ Mock data created successfully")


def test_graph_loading():
    """Test loading preprocessed data"""
    print("🧪 Testing graph data loading...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    data_dir = "data/processed/gnn_ready"

    # Create mock data if real data doesn't exist
    if not os.path.exists(data_dir):
        create_mock_data(data_dir)

    try:
        constructor = MusicGraphConstructor(data_dir)
        constructor.load_preprocessed_data()

        # Check that all required data is loaded
        assert constructor.mappings is not None, "Mappings not loaded"
        assert constructor.entity_counts is not None, "Entity counts not loaded"
        assert constructor.edges is not None, "Edges not loaded"
        assert constructor.features is not None, "Features not loaded"
        assert constructor.splits is not None, "Splits not loaded"

        print("✅ Data loading test passed")
        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_graph_construction():
    """Test graph construction"""
    print("\n🧪 Testing graph construction...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    data_dir = "data/processed/gnn_ready"

    # Create mock data if real data doesn't exist
    if not os.path.exists(data_dir):
        create_mock_data(data_dir)

    try:
        constructor = MusicGraphConstructor(data_dir)
        constructor.load_preprocessed_data()
        constructor.create_node_mappings()
        constructor.create_full_graph()

        # Check graph properties
        graph = constructor.full_graph
        assert graph is not None, "Graph not created"
        assert len(graph.node_types) > 0, "No node types in graph"
        assert len(graph.edge_types) > 0, "No edge types in graph"

        # Check that all node types have features
        for node_type in graph.node_types:
            assert hasattr(graph[node_type], 'x'), f"Node type {node_type} missing features"
            assert graph[node_type].x is not None, f"Node type {node_type} has None features"

        print("✅ Graph construction test passed")
        return True

    except Exception as e:
        print(f"❌ Graph construction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_val_test_splits():
    """Test train/validation/test graph creation"""
    print("\n🧪 Testing train/val/test splits...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    data_dir = "data/processed/gnn_ready"

    # Create mock data if real data doesn't exist
    if not os.path.exists(data_dir):
        create_mock_data(data_dir)

    try:
        constructor = MusicGraphConstructor(data_dir)
        constructor.load_preprocessed_data()
        constructor.create_node_mappings()
        constructor.create_full_graph()
        constructor.create_train_val_test_graphs()

        # Check all graphs exist
        assert constructor.train_graph is not None, "Training graph not created"
        assert constructor.val_graph is not None, "Validation graph not created"
        assert constructor.test_graph is not None, "Test graph not created"

        # Check that all graphs have the same node types
        node_types = constructor.full_graph.node_types
        assert constructor.train_graph.node_types == node_types, "Train graph node types mismatch"
        assert constructor.val_graph.node_types == node_types, "Val graph node types mismatch"
        assert constructor.test_graph.node_types == node_types, "Test graph node types mismatch"

        print("✅ Train/val/test splits test passed")
        return True

    except Exception as e:
        print(f"❌ Train/val/test splits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_utils():
    """Test graph utilities"""
    print("\n🧪 Testing graph utilities...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    try:
        # Create a simple test graph
        test_graph = HeteroData()

        # Add node features
        test_graph['user'].x = torch.randn(100, 4)
        test_graph['item'].x = torch.randn(50, 6)

        # Add edges
        edge_index = torch.randint(0, 100, (2, 200))
        edge_index[1] = torch.randint(0, 50, (200,))  # Ensure valid dst indices
        test_graph['user', 'likes', 'item'].edge_index = edge_index

        # Test utility functions
        info = GraphUtils.get_graph_info(test_graph)
        assert info['total_nodes'] == 150, "Incorrect node count"
        assert info['total_edges'] == 200, "Incorrect edge count"

        # Test positive edge extraction
        pos_edges = GraphUtils.get_positive_edges(test_graph, ('user', 'likes', 'item'))
        assert pos_edges.shape == (200, 2), "Incorrect positive edges shape"

        # Test negative edge sampling
        neg_edges = GraphUtils.sample_negative_edges(test_graph, ('user', 'likes', 'item'), 100)
        assert neg_edges.shape[0] <= 100, "Too many negative edges sampled"
        assert neg_edges.shape[1] == 2, "Incorrect negative edges shape"

        # Test consistency check
        consistency = GraphUtils.check_graph_consistency(test_graph)
        assert all(consistency.values()), "Graph consistency check failed"

        print("✅ Graph utilities test passed")
        return True

    except Exception as e:
        print(f"❌ Graph utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_link_prediction_data():
    """Test link prediction data preparation"""
    print("\n🧪 Testing link prediction data preparation...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    try:
        # Create a test graph
        test_graph = HeteroData()
        test_graph['user'].x = torch.randn(50, 4)
        test_graph['item'].x = torch.randn(30, 6)

        edge_index = torch.randint(0, 50, (2, 100))
        edge_index[1] = torch.randint(0, 30, (100,))
        test_graph['user', 'likes', 'item'].edge_index = edge_index

        # Prepare link prediction data
        link_data = prepare_link_prediction_data(
            test_graph,
            ('user', 'likes', 'item'),
            neg_sampling_ratio=1.0
        )

        assert 'pos_edges' in link_data, "Missing positive edges"
        assert 'neg_edges' in link_data, "Missing negative edges"
        assert link_data['pos_edges'].shape[0] == 100, "Incorrect number of positive edges"
        assert link_data['neg_edges'].shape[0] <= 100, "Too many negative edges"

        print("✅ Link prediction data test passed")
        return True

    except Exception as e:
        print(f"❌ Link prediction data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load_graphs():
    """Test saving and loading graphs"""
    print("\n🧪 Testing graph save/load functionality...")

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric not available, skipping test")
        return False

    data_dir = "data/processed/gnn_ready"

    # Create mock data if real data doesn't exist
    if not os.path.exists(data_dir):
        create_mock_data(data_dir)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Construct and save graphs
        constructor = MusicGraphConstructor(data_dir)
        graphs = constructor.run_complete_pipeline(temp_dir)

        # Check files were saved
        expected_files = ['full_graph.pt', 'train_graph.pt', 'val_graph.pt', 'test_graph.pt']
        for filename in expected_files:
            file_path = os.path.join(temp_dir, filename)
            assert os.path.exists(file_path), f"File not saved: {filename}"

        # Test loading
        loaded_graphs = GraphUtils.load_graphs(temp_dir)
        assert len(loaded_graphs) == 4, "Not all graphs loaded"

        # Compare original and loaded graphs
        for name in graphs.keys():
            if name in loaded_graphs and graphs[name] is not None:
                original = graphs[name]
                loaded = loaded_graphs[name]

                # Check node types match
                assert original.node_types == loaded.node_types, f"Node types mismatch for {name}"

                # Check node features match
                for node_type in original.node_types:
                    if hasattr(original[node_type], 'x'):
                        assert torch.allclose(original[node_type].x, loaded[node_type].x), \
                            f"Features mismatch for {node_type} in {name}"

        print("✅ Save/load test passed")
        return True

    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests and report results"""
    print("🧪 RUNNING GRAPH CONSTRUCTION TESTS")
    print("=" * 50)

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("\n❌ CRITICAL: PyTorch Geometric is not installed!")
        print("Please install it using one of these commands:")
        print("   pip install torch-geometric")
        print("   conda install pyg -c pyg")
        print("\nOr follow the installation guide at:")
        print("   https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html")
        return False

    tests = [
        ("Data Loading", test_graph_loading),
        ("Graph Construction", test_graph_construction),
        ("Train/Val/Test Splits", test_train_val_test_splits),
        ("Graph Utilities", test_graph_utils),
        ("Link Prediction Data", test_link_prediction_data),
        ("Save/Load Graphs", test_save_load_graphs)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Graph construction is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)