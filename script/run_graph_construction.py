#!/usr/bin/env python3
"""
Script to run graph construction for the music recommendation GNN project.

This script converts preprocessed data into PyTorch Geometric graphs ready for training.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_construction import MusicGraphConstructor, load_all_graphs


def main():
    parser = argparse.ArgumentParser(description='Build heterogeneous graphs for music recommendation')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/gnn_ready',
        help='Directory containing preprocessed data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/graphs',
        help='Directory to save constructed graphs'
    )
    parser.add_argument(
        '--load_only',
        action='store_true',
        help='Load existing graphs instead of constructing new ones'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing graphs without reconstruction'
    )

    args = parser.parse_args()

    # Convert to absolute paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("🎵 MUSIC GNN GRAPH CONSTRUCTION")
    print("=" * 50)
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()

    if args.load_only:
        print("📥 Loading existing graphs...")
        try:
            graphs = load_all_graphs(str(output_dir))
            print(f"\n✅ Successfully loaded {len(graphs)} graphs")

            # Print graph info
            for name, graph in graphs.items():
                print(f"   • {name}: {len(graph.node_types)} node types, {len(graph.edge_types)} edge types")

        except Exception as e:
            print(f"❌ Error loading graphs: {e}")
            sys.exit(1)

    elif args.verify_only:
        print("🔍 Verifying existing graphs...")
        try:
            graphs = load_all_graphs(str(output_dir))

            # Basic verification
            required_graphs = ['full_graph', 'train_graph', 'val_graph', 'test_graph']
            for graph_name in required_graphs:
                if graph_name in graphs:
                    graph = graphs[graph_name]
                    print(f"✅ {graph_name}: {len(graph.node_types)} nodes, {len(graph.edge_types)} edges")
                else:
                    print(f"❌ Missing: {graph_name}")

        except Exception as e:
            print(f"❌ Error verifying graphs: {e}")
            sys.exit(1)

    else:
        # Construct new graphs
        try:
            # Check if data directory exists
            if not data_dir.exists():
                print(f"❌ Data directory not found: {data_dir}")
                print("   Please run data preprocessing first.")
                sys.exit(1)

            # Initialize constructor
            constructor = MusicGraphConstructor(str(data_dir))

            # Run complete pipeline
            graphs = constructor.run_complete_pipeline(str(output_dir))

            print(f"\n🎉 Graph construction completed successfully!")
            print(f"📁 Graphs saved to: {output_dir}")

            # Print final summary
            print(f"\n📊 FINAL SUMMARY:")
            for name, graph in graphs.items():
                if graph is not None:
                    total_nodes = sum(graph[node_type].x.shape[0] for node_type in graph.node_types)
                    total_edges = sum(graph[edge_type].edge_index.shape[1] for edge_type in graph.edge_types)
                    print(f"   • {name}: {total_nodes:,} nodes, {total_edges:,} edges")

        except KeyboardInterrupt:
            print("\n⚠️  Graph construction interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during graph construction: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n🚀 Ready for GNN model training!")


if __name__ == "__main__":
    main()