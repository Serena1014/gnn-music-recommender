#!/usr/bin/env python3
"""
Test Script for Data Preprocessing Pipeline

This script tests the data preprocessing pipeline with a small sample
to ensure everything works correctly before running on the full dataset.

Usage:
    python test_preprocessing.py
"""

import json
import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_test_data():
    """Create a small test dataset"""
    test_playlists = []

    # Create 10 test playlists
    for pid in range(10):
        playlist = {
            'pid': pid,
            'name': f'Test Playlist {pid}',
            'collaborative': pid % 3 == 0,  # Every 3rd playlist is collaborative
            'modified_at': 1500000000 + pid * 1000,
            'num_followers': pid * 5,
            'tracks': []
        }

        # Add 5-15 tracks per playlist
        num_tracks = 5 + (pid % 10)
        for track_idx in range(num_tracks):
            track = {
                'track_uri': f'spotify:track:test_track_{(pid * 20 + track_idx) % 50}',  # 50 unique tracks
                'artist_uri': f'spotify:artist:test_artist_{(pid + track_idx) % 20}',  # 20 unique artists
                'album_uri': f'spotify:album:test_album_{(pid + track_idx) % 30}',  # 30 unique albums
                'track_name': f'Test Track {track_idx}',
                'artist_name': f'Test Artist {(pid + track_idx) % 20}',
                'album_name': f'Test Album {(pid + track_idx) % 30}',
                'duration_ms': 180000 + (track_idx * 15000)  # 3-6 minutes
            }
            playlist['tracks'].append(track)

        test_playlists.append(playlist)

    # Create test data structure
    test_data = {
        'playlists': test_playlists,
        'info': {
            'sampling_method': 'test_data_generation',
            'generated_at': '2025-06-25T00:00:00'
        },
        'sampling_stats': {
            'total_playlists': 10,
            'total_tracks': sum(len(p['tracks']) for p in test_playlists)
        }
    }

    return test_data


def run_preprocessing_test():
    """Run preprocessing on test data"""
    print("🧪 TESTING DATA PREPROCESSING PIPELINE")
    print("=" * 50)

    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "test_input.json")
    output_dir = os.path.join(temp_dir, "test_output")

    try:
        # 1. Create test data
        print("📝 Creating test data...")
        test_data = create_test_data()

        with open(input_file, 'w') as f:
            json.dump(test_data, f)

        print(f"✅ Test data created: {len(test_data['playlists'])} playlists")

        # 2. Run preprocessing
        print("\n🔄 Running preprocessing...")

        # Import and run preprocessing
        from data_preprocessing import (
            DataLoader, EntityMapper, GraphBuilder,
            FeatureExtractor, DataSplitter, save_preprocessed_data,
            verify_preprocessed_data
        )

        # Load data
        playlists, metadata = DataLoader.load_sampled_data(input_file)
        data_stats = DataLoader.explore_data_structure(playlists)

        # Create mappings
        mapper = EntityMapper()
        mappings = mapper.create_mappings(playlists)

        # Build graph
        graph_builder = GraphBuilder(mappings, mapper.entity_counts)
        edges = graph_builder.build_edges(playlists)

        # Extract features
        feature_extractor = FeatureExtractor(mappings, mapper.entity_counts, mapper.reverse_mappings)
        features = feature_extractor.extract_all_features(playlists)

        # Create splits
        splitter = DataSplitter(edges, mappings)
        splits = splitter.create_playlist_track_splits(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        # Create negative samples
        negative_val = splitter.create_negative_samples(
            splits['val_edges'],
            mapper.entity_counts['playlists'],
            mapper.entity_counts['tracks'],
            num_negative=len(splits['val_edges']),
            random_seed=42
        )

        negative_test = splitter.create_negative_samples(
            splits['test_edges'],
            mapper.entity_counts['playlists'],
            mapper.entity_counts['tracks'],
            num_negative=len(splits['test_edges']),
            random_seed=42
        )

        # Save data
        save_preprocessed_data(
            output_dir=output_dir,
            mappings=mappings,
            edges=edges,
            features=features,
            splits=splits,
            negative_val=negative_val,
            negative_test=negative_test,
            entity_counts=mapper.entity_counts,
            reverse_mappings=mapper.reverse_mappings
        )

        print("✅ Preprocessing completed successfully!")

        # 3. Verify results
        print("\n🔍 Verifying results...")
        verify_preprocessed_data(output_dir)

        # 4. Test data loader
        print("\n📊 Testing data loader...")
        from utils.data_loader import GNNDataLoader

        loader = GNNDataLoader(output_dir)
        data = loader.load_all_data()
        loader.print_data_summary()

        # Basic validation
        assert len(data['edges']['playlist_track']) > 0, "No playlist-track edges found"
        assert data['features']['playlist'].shape[0] == data['entity_counts'][
            'playlists'], "Playlist feature count mismatch"
        assert len(data['splits']['train_edges']) > 0, "No training edges found"

        print("\n✅ All tests passed!")

        # 5. Print test results
        print("\n📈 TEST RESULTS SUMMARY:")
        print(f"   • Input playlists: {len(test_data['playlists'])}")
        print(f"   • Unique entities: {sum(data['entity_counts'].values())}")
        print(f"   • Total edges: {sum(len(edges) for edges in data['edges'].values())}")
        print(f"   • Training edges: {len(data['splits']['train_edges'])}")
        print(f"   • Output files: {len(os.listdir(output_dir))}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the test"""
    success = run_preprocessing_test()

    if success:
        print("\n🎉 Data preprocessing pipeline test completed successfully!")
        print("   You can now run the full preprocessing on your dataset.")
        return 0
    else:
        print("\n💥 Test failed! Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())