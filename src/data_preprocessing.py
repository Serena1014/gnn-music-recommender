#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for GNN Music Recommender System

This script transforms sampled Spotify playlist data into GNN-ready format with:
- Entity mappings (playlists, tracks, artists, albums, users)
- Graph edges and relationships
- Node features for all entity types
- Train/validation/test splits (70/15/15)
- Negative sampling for evaluation

Usage:
    python data_preprocessing.py --input_file path/to/sampled_data.json --output_dir path/to/output
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle
import os
import argparse
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and explore sampled dataset"""

    @staticmethod
    def load_sampled_data(file_path: str) -> Tuple[List[Dict], Dict]:
        """Load the sampled dataset"""
        logger.info(f"📂 Loading sampled data from: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        playlists = data.get('playlists', [])
        info = data.get('info', {})
        sampling_stats = data.get('sampling_stats', {})

        logger.info(f"✅ Loaded {len(playlists):,} playlists")
        logger.info(f"📊 Sampling method: {info.get('sampling_method', 'unknown')}")

        return playlists, {'info': info, 'sampling_stats': sampling_stats}

    @staticmethod
    def explore_data_structure(playlists: List[Dict]) -> Dict:
        """Explore the structure of sampled data"""
        logger.info("🔍 EXPLORING DATA STRUCTURE")
        logger.info("=" * 40)

        # Basic statistics
        total_playlists = len(playlists)
        playlist_lengths = [len(p.get('tracks', [])) for p in playlists]

        # Extract all entities
        all_tracks = set()
        all_artists = set()
        all_albums = set()
        user_playlist_count = Counter()

        for playlist in playlists:
            # Extract user ID (simplified)
            name = playlist.get('name', '').strip()
            user_id = name.split()[0] if name else f"user_{playlist.get('pid', 0) % 1000}"
            user_playlist_count[user_id] += 1

            # Extract track info
            tracks = playlist.get('tracks', [])
            for track in tracks:
                track_uri = track.get('track_uri', '')
                artist_uri = track.get('artist_uri', '')
                album_uri = track.get('album_uri', '')

                if track_uri:
                    all_tracks.add(track_uri)
                if artist_uri:
                    all_artists.add(artist_uri)
                if album_uri:
                    all_albums.add(album_uri)

        stats = {
            'num_playlists': total_playlists,
            'num_unique_tracks': len(all_tracks),
            'num_unique_artists': len(all_artists),
            'num_unique_albums': len(all_albums),
            'num_unique_users': len(user_playlist_count),
            'avg_playlist_length': np.mean(playlist_lengths),
            'min_playlist_length': min(playlist_lengths),
            'max_playlist_length': max(playlist_lengths),
            'total_track_occurrences': sum(playlist_lengths)
        }

        # Print statistics
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   • Playlists: {stats['num_playlists']:,}")
        logger.info(f"   • Unique tracks: {stats['num_unique_tracks']:,}")
        logger.info(f"   • Unique artists: {stats['num_unique_artists']:,}")
        logger.info(f"   • Unique albums: {stats['num_unique_albums']:,}")
        logger.info(f"   • Unique users: {stats['num_unique_users']:,}")
        logger.info(f"   • Avg playlist length: {stats['avg_playlist_length']:.1f}")
        logger.info(f"   • Playlist length range: {stats['min_playlist_length']}-{stats['max_playlist_length']}")
        logger.info(f"   • Total track occurrences: {stats['total_track_occurrences']:,}")

        return stats


class EntityMapper:
    """Create bidirectional mappings between entities and integer IDs"""

    def __init__(self):
        self.mappings = {}
        self.reverse_mappings = {}
        self.entity_counts = {}

    def create_mappings(self, playlists: List[Dict]) -> Dict:
        """Create mappings for all entities"""
        logger.info("🗺️  CREATING ENTITY MAPPINGS")
        logger.info("=" * 40)

        # Collect all entities
        entities = {
            'playlists': {},
            'tracks': {},
            'artists': {},
            'albums': {},
            'users': {}
        }

        # Extract entities from playlists
        logger.info("📊 Extracting entities...")

        for playlist in playlists:
            # Playlist ID
            pid = playlist.get('pid')
            if pid is not None:
                entities['playlists'][str(pid)] = playlist

            # User ID (simplified extraction)
            name = playlist.get('name', '').strip()
            user_id = name.split()[0] if name else f"user_{pid % 1000}"
            entities['users'][user_id] = entities['users'].get(user_id, 0) + 1

            # Track, artist, album info
            tracks = playlist.get('tracks', [])
            for track in tracks:
                track_uri = track.get('track_uri', '')
                artist_uri = track.get('artist_uri', '')
                album_uri = track.get('album_uri', '')

                if track_uri:
                    entities['tracks'][track_uri] = track
                if artist_uri:
                    entities['artists'][artist_uri] = {
                        'artist_name': track.get('artist_name', ''),
                        'artist_uri': artist_uri
                    }
                if album_uri:
                    entities['albums'][album_uri] = {
                        'album_name': track.get('album_name', ''),
                        'album_uri': album_uri
                    }

        # Create integer mappings
        logger.info("🔢 Creating integer ID mappings...")

        for entity_type, entity_dict in entities.items():
            entity_list = list(entity_dict.keys())

            # Create forward mapping (entity -> int)
            self.mappings[entity_type] = {
                entity: idx for idx, entity in enumerate(entity_list)
            }

            # Create reverse mapping (int -> entity)
            self.reverse_mappings[entity_type] = {
                idx: entity for entity, idx in self.mappings[entity_type].items()
            }

            # Store counts
            self.entity_counts[entity_type] = len(entity_list)

            logger.info(f"   ✅ {entity_type}: {len(entity_list):,} entities")

        return self.mappings

    def get_mapping_stats(self) -> Dict:
        """Get statistics about the mappings"""
        return {
            'entity_counts': self.entity_counts,
            'total_nodes': sum(self.entity_counts.values())
        }


class GraphBuilder:
    """Build edges for the music recommendation graph"""

    def __init__(self, mappings: Dict, entity_counts: Dict):
        self.mappings = mappings
        self.entity_counts = entity_counts
        self.edges = {}

    def build_edges(self, playlists: List[Dict]) -> Dict:
        """Build all edge types for the graph"""
        logger.info("🔗 BUILDING GRAPH EDGES")
        logger.info("=" * 40)

        # Initialize edge lists
        self.edges = {
            'playlist_track': [],  # Playlist contains track
            'track_artist': [],  # Track by artist
            'track_album': [],  # Track in album
            'user_playlist': [],  # User created playlist
            'playlist_user': []  # Reverse of user_playlist
        }

        logger.info("🔗 Extracting relationships...")

        for playlist in playlists:
            pid = playlist.get('pid')
            playlist_name = playlist.get('name', '').strip()

            # Get mapped IDs
            playlist_id = self.mappings['playlists'].get(str(pid))
            if playlist_id is None:
                continue

            # Extract user
            user_name = playlist_name.split()[0] if playlist_name else f"user_{pid % 1000}"
            user_id = self.mappings['users'].get(user_name)

            # User-Playlist edges
            if user_id is not None:
                self.edges['user_playlist'].append([user_id, playlist_id])
                self.edges['playlist_user'].append([playlist_id, user_id])

            # Process tracks
            tracks = playlist.get('tracks', [])
            for track in tracks:
                track_uri = track.get('track_uri', '')
                artist_uri = track.get('artist_uri', '')
                album_uri = track.get('album_uri', '')

                track_id = self.mappings['tracks'].get(track_uri)

                if track_id is not None:
                    # Playlist-Track edge
                    self.edges['playlist_track'].append([playlist_id, track_id])

                    # Track-Artist edge
                    artist_id = self.mappings['artists'].get(artist_uri)
                    if artist_id is not None:
                        self.edges['track_artist'].append([track_id, artist_id])

                    # Track-Album edge
                    album_id = self.mappings['albums'].get(album_uri)
                    if album_id is not None:
                        self.edges['track_album'].append([track_id, album_id])

        # Convert to numpy arrays and remove duplicates
        logger.info("🔧 Processing edge lists...")

        for edge_type, edge_list in self.edges.items():
            if edge_list:
                # Convert to numpy array
                edges_array = np.array(edge_list)

                # Remove duplicates
                unique_edges = np.unique(edges_array, axis=0)

                self.edges[edge_type] = unique_edges

                logger.info(f"   ✅ {edge_type}: {len(unique_edges):,} edges")
            else:
                self.edges[edge_type] = np.array([]).reshape(0, 2)
                logger.info(f"   ⚠️  {edge_type}: 0 edges")

        return self.edges

    def get_graph_statistics(self) -> Dict:
        """Calculate graph statistics"""
        stats = {}

        for edge_type, edges in self.edges.items():
            stats[edge_type] = {
                'num_edges': len(edges),
                'density': len(edges) / (self.entity_counts.get('playlists', 1) * self.entity_counts.get('tracks',
                                                                                                         1)) if 'playlist' in edge_type and 'track' in edge_type else 0
            }

        return stats


class FeatureExtractor:
    """Extract features for graph nodes"""

    def __init__(self, mappings: Dict, entity_counts: Dict, reverse_mappings: Dict):
        self.mappings = mappings
        self.entity_counts = entity_counts
        self.reverse_mappings = reverse_mappings
        self.features = {}

    def extract_playlist_features(self, playlists: List[Dict]) -> np.ndarray:
        """Extract features for playlist nodes"""
        logger.info("🎵 Extracting playlist features...")

        num_playlists = self.entity_counts['playlists']

        # Create mapping from PID to playlist data
        pid_to_playlist = {str(p.get('pid')): p for p in playlists}

        features = []

        for playlist_idx in range(num_playlists):
            # Get original playlist ID
            original_pid = self.reverse_mappings['playlists'][playlist_idx]
            playlist_data = pid_to_playlist.get(original_pid, {})

            # Extract features
            num_tracks = len(playlist_data.get('tracks', []))
            num_followers = playlist_data.get('num_followers', 0)
            is_collaborative = 1 if playlist_data.get('collaborative', False) else 0

            # Temporal features
            modified_at = playlist_data.get('modified_at', 0)
            normalized_time = (modified_at - 1400000000) / 100000000 if modified_at > 0 else 0

            # Text features
            name = playlist_data.get('name', '')
            has_name = 1 if len(name.strip()) > 0 else 0
            name_length = len(name.strip())

            # Combine features
            playlist_features = [
                num_tracks,
                np.log1p(num_followers),  # Log transform followers
                is_collaborative,
                normalized_time,
                has_name,
                name_length
            ]

            features.append(playlist_features)

        features_array = np.array(features, dtype=np.float32)
        logger.info(f"   ✅ Playlist features shape: {features_array.shape}")

        return features_array

    def extract_track_features(self, playlists: List[Dict]) -> np.ndarray:
        """Extract features for track nodes"""
        logger.info("🎼 Extracting track features...")

        num_tracks = self.entity_counts['tracks']

        # Track statistics
        track_stats = defaultdict(lambda: {
            'playlist_count': 0,
            'total_position': 0,
            'positions': [],
            'durations': []
        })

        # Collect track statistics
        for playlist in playlists:
            tracks = playlist.get('tracks', [])
            for pos, track in enumerate(tracks):
                track_uri = track.get('track_uri', '')
                duration = track.get('duration_ms', 0)

                if track_uri:
                    track_stats[track_uri]['playlist_count'] += 1
                    track_stats[track_uri]['total_position'] += pos
                    track_stats[track_uri]['positions'].append(pos)
                    if duration > 0:
                        track_stats[track_uri]['durations'].append(duration)

        # Create feature matrix
        features = []

        for track_idx in range(num_tracks):
            # Get original track URI
            track_uri = self.reverse_mappings['tracks'][track_idx]
            stats = track_stats[track_uri]

            # Extract features
            playlist_count = stats['playlist_count']
            avg_position = stats['total_position'] / max(playlist_count, 1)
            position_std = np.std(stats['positions']) if stats['positions'] else 0
            avg_duration = np.mean(stats['durations']) if stats['durations'] else 180000  # Default 3 min

            track_features = [
                np.log1p(playlist_count),  # Log of popularity
                avg_position,  # Average position in playlists
                position_std,  # Position variability
                avg_duration / 60000,  # Duration in minutes
            ]

            features.append(track_features)

        features_array = np.array(features, dtype=np.float32)
        logger.info(f"   ✅ Track features shape: {features_array.shape}")

        return features_array

    def extract_user_features(self, playlists: List[Dict]) -> np.ndarray:
        """Extract features for user nodes"""
        logger.info("👥 Extracting user features...")

        num_users = self.entity_counts['users']

        # User statistics
        user_stats = defaultdict(lambda: {
            'playlist_count': 0,
            'total_tracks': 0,
            'unique_tracks': set(),
            'collaborative_count': 0
        })

        # Collect user statistics
        for playlist in playlists:
            name = playlist.get('name', '').strip()
            user_name = name.split()[0] if name else f"user_{playlist.get('pid', 0) % 1000}"

            tracks = playlist.get('tracks', [])
            is_collaborative = playlist.get('collaborative', False)

            user_stats[user_name]['playlist_count'] += 1
            user_stats[user_name]['total_tracks'] += len(tracks)
            user_stats[user_name]['unique_tracks'].update([t.get('track_uri', '') for t in tracks])
            if is_collaborative:
                user_stats[user_name]['collaborative_count'] += 1

        # Create feature matrix
        features = []

        for user_idx in range(num_users):
            # Get original user name
            user_name = self.reverse_mappings['users'][user_idx]
            stats = user_stats[user_name]

            # Extract features
            playlist_count = stats['playlist_count']
            avg_playlist_length = stats['total_tracks'] / max(playlist_count, 1)
            unique_track_count = len(stats['unique_tracks'])
            collaborative_ratio = stats['collaborative_count'] / max(playlist_count, 1)

            user_features = [
                np.log1p(playlist_count),  # Log of activity level
                avg_playlist_length,  # Average playlist length
                np.log1p(unique_track_count),  # Log of music diversity
                collaborative_ratio  # Collaboration tendency
            ]

            features.append(user_features)

        features_array = np.array(features, dtype=np.float32)
        logger.info(f"   ✅ User features shape: {features_array.shape}")

        return features_array

    def extract_all_features(self, playlists: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract features for all node types"""
        logger.info("🎨 EXTRACTING NODE FEATURES")
        logger.info("=" * 40)

        features = {}

        # Extract features for each node type
        features['playlist'] = self.extract_playlist_features(playlists)
        features['track'] = self.extract_track_features(playlists)
        features['user'] = self.extract_user_features(playlists)

        # Simple features for artists and albums (placeholder)
        np.random.seed(42)  # For reproducibility
        features['artist'] = np.random.randn(self.entity_counts['artists'], 4).astype(np.float32)
        features['album'] = np.random.randn(self.entity_counts['albums'], 4).astype(np.float32)

        logger.info(f"   ⚠️  Artist/Album features: Using random placeholders")

        return features


class DataSplitter:
    """Create train/validation/test splits for the recommendation task"""

    def __init__(self, edges: Dict, mappings: Dict):
        self.edges = edges
        self.mappings = mappings

    def create_playlist_track_splits(self, train_ratio: float = 0.7,
                                     val_ratio: float = 0.15,
                                     test_ratio: float = 0.15,
                                     random_seed: int = 42) -> Dict:
        """Create splits for playlist-track edges with 70/15/15 ratio"""
        logger.info("✂️  CREATING TRAIN/VALIDATION/TEST SPLITS (70/15/15)")
        logger.info("=" * 50)

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Get playlist-track edges
        playlist_track_edges = self.edges['playlist_track']
        num_edges = len(playlist_track_edges)

        logger.info(f"📊 Total playlist-track edges: {num_edges:,}")
        logger.info(f"📊 Split ratios: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")

        # Shuffle edges
        indices = np.random.permutation(num_edges)

        # Calculate split sizes
        train_size = int(num_edges * train_ratio)
        val_size = int(num_edges * val_ratio)
        test_size = num_edges - train_size - val_size

        logger.info(f"📈 Calculated split sizes:")
        logger.info(f"   • Train: {train_size:,} edges ({train_size / num_edges:.1%})")
        logger.info(f"   • Validation: {val_size:,} edges ({val_size / num_edges:.1%})")
        logger.info(f"   • Test: {test_size:,} edges ({test_size / num_edges:.1%})")

        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = {
            'train_edges': playlist_track_edges[train_indices],
            'val_edges': playlist_track_edges[val_indices],
            'test_edges': playlist_track_edges[test_indices],
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }

        logger.info(f"✅ Final split sizes:")
        logger.info(f"   • Train edges: {len(splits['train_edges']):,}")
        logger.info(f"   • Validation edges: {len(splits['val_edges']):,}")
        logger.info(f"   • Test edges: {len(splits['test_edges']):,}")

        return splits

    def create_negative_samples(self, positive_edges: np.ndarray,
                                num_playlists: int, num_tracks: int,
                                num_negative: int = None,
                                random_seed: int = 42) -> np.ndarray:
        """Create negative samples for link prediction"""
        np.random.seed(random_seed)

        if num_negative is None:
            num_negative = len(positive_edges)

        # Create set of positive edges for efficient lookup
        positive_set = set(map(tuple, positive_edges))

        # Sample negative edges
        negative_edges = []
        max_attempts = num_negative * 10  # Prevent infinite loop
        attempts = 0

        while len(negative_edges) < num_negative and attempts < max_attempts:
            # Random playlist and track
            playlist_id = np.random.randint(0, num_playlists)
            track_id = np.random.randint(0, num_tracks)

            # Check if this is not a positive edge
            if (playlist_id, track_id) not in positive_set:
                negative_edges.append([playlist_id, track_id])

            attempts += 1

        return np.array(negative_edges)


def save_preprocessed_data(output_dir: str, mappings: Dict, edges: Dict,
                           features: Dict, splits: Dict,
                           negative_val: np.ndarray, negative_test: np.ndarray,
                           entity_counts: Dict, reverse_mappings: Dict) -> str:
    """Save all preprocessed data"""
    logger.info("💾 SAVING PREPROCESSED DATA")
    logger.info("=" * 40)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save mappings
    with open(f"{output_dir}/mappings.pkl", 'wb') as f:
        pickle.dump(mappings, f)
    logger.info(f"✅ Saved mappings to {output_dir}/mappings.pkl")

    # Save reverse mappings
    with open(f"{output_dir}/reverse_mappings.pkl", 'wb') as f:
        pickle.dump(reverse_mappings, f)
    logger.info(f"✅ Saved reverse mappings to {output_dir}/reverse_mappings.pkl")

    # Save entity counts
    with open(f"{output_dir}/entity_counts.pkl", 'wb') as f:
        pickle.dump(entity_counts, f)
    logger.info(f"✅ Saved entity counts to {output_dir}/entity_counts.pkl")

    # Save edges
    np.savez(f"{output_dir}/edges.npz", **edges)
    logger.info(f"✅ Saved edges to {output_dir}/edges.npz")

    # Save features
    np.savez(f"{output_dir}/features.npz", **features)
    logger.info(f"✅ Saved features to {output_dir}/features.npz")

    # Save splits
    splits_with_negatives = {
        **splits,
        'negative_val': negative_val,
        'negative_test': negative_test
    }
    # Convert any non-array values to ensure compatibility
    splits_to_save = {}
    for key, value in splits_with_negatives.items():
        if isinstance(value, np.ndarray):
            splits_to_save[key] = value
        elif isinstance(value, dict):
            # Convert dict to a format that can be saved
            splits_to_save[key] = np.array([value], dtype=object)
        else:
            splits_to_save[key] = np.array([value])

    np.savez(f"{output_dir}/splits.npz", **splits_to_save)
    logger.info(f"✅ Saved splits to {output_dir}/splits.npz")

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'entity_counts': entity_counts,
        'feature_dimensions': {k: v.shape for k, v in features.items()},
        'edge_counts': {k: len(v) for k, v in edges.items()},
        'split_sizes': {
            'train': len(splits['train_edges']),
            'val': len(splits['val_edges']),
            'test': len(splits['test_edges'])
        },
        'split_ratios': splits['split_ratios'],
        'preprocessing_notes': {
            'split_strategy': '70/15/15 ratio following Graph ML best practices',
            'negative_sampling': '1:1 positive to negative ratio',
            'edge_type_focus': 'playlist-track edges for recommendation task',
            'random_seed': 42
        }
    }

    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata to {output_dir}/metadata.json")

    logger.info(f"\n🎉 All preprocessed data saved to: {output_dir}")
    return output_dir


def verify_preprocessed_data(data_dir: str):
    """Verify the preprocessed data"""
    logger.info("✅ VERIFICATION SUMMARY")
    logger.info("=" * 40)

    # Load and check each file
    files_to_check = [
        'mappings.pkl',
        'reverse_mappings.pkl',
        'entity_counts.pkl',
        'edges.npz',
        'features.npz',
        'splits.npz',
        'metadata.json'
    ]

    for file_name in files_to_check:
        file_path = f"{data_dir}/{file_name}"
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            logger.info(f"✅ {file_name}: {file_size:.2f} MB")
        else:
            logger.error(f"❌ {file_name}: Missing!")

    # Load metadata and print summary
    try:
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)

        logger.info(f"\n📊 PREPROCESSING SUMMARY:")
        logger.info(f"   🎵 Playlists: {metadata['entity_counts']['playlists']:,}")
        logger.info(f"   🎼 Tracks: {metadata['entity_counts']['tracks']:,}")
        logger.info(f"   🎤 Artists: {metadata['entity_counts']['artists']:,}")
        logger.info(f"   💿 Albums: {metadata['entity_counts']['albums']:,}")
        logger.info(f"   👥 Users: {metadata['entity_counts']['users']:,}")
        logger.info(f"   🔗 Total edges: {sum(metadata['edge_counts'].values()):,}")
        logger.info(
            f"   📚 Training edges: {metadata['split_sizes']['train']:,} ({metadata['split_ratios']['train']:.0%})")
        logger.info(
            f"   🔍 Validation edges: {metadata['split_sizes']['val']:,} ({metadata['split_ratios']['val']:.0%})")
        logger.info(f"   🧪 Test edges: {metadata['split_sizes']['test']:,} ({metadata['split_ratios']['test']:.0%})")

    except Exception as e:
        logger.error(f"⚠️  Could not load metadata: {e}")


def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='Preprocess data for GNN music recommender')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input sampled JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    logger.info("🎵 GNN DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("🎯 Goal: Transform sampled playlists into GNN-ready graph structure")
    logger.info(f"📊 Input: {args.input_file}")
    logger.info(f"📈 Output: {args.output_dir}")
    logger.info(f"📋 Split Ratio: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / {args.test_ratio:.0%} test")

    # 1. Load data
    playlists, metadata = DataLoader.load_sampled_data(args.input_file)
    data_stats = DataLoader.explore_data_structure(playlists)

    # 2. Create entity mappings
    mapper = EntityMapper()
    mappings = mapper.create_mappings(playlists)
    mapping_stats = mapper.get_mapping_stats()
    logger.info(f"📈 Total graph nodes: {mapping_stats['total_nodes']:,}")

    # 3. Build graph edges
    graph_builder = GraphBuilder(mappings, mapper.entity_counts)
    edges = graph_builder.build_edges(playlists)
    graph_stats = graph_builder.get_graph_statistics()

    # Print graph statistics
    logger.info("📊 Graph Statistics:")
    for edge_type, stats in graph_stats.items():
        logger.info(f"   • {edge_type}: {stats['num_edges']:,} edges")

    # 4. Extract node features
    feature_extractor = FeatureExtractor(mappings, mapper.entity_counts, mapper.reverse_mappings)
    node_features = feature_extractor.extract_all_features(playlists)

    # 5. Create data splits
    splitter = DataSplitter(edges, mappings)
    splits = splitter.create_playlist_track_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    # Create negative samples for evaluation
    logger.info("🔄 Creating negative samples for evaluation...")
    negative_val = splitter.create_negative_samples(
        splits['val_edges'],
        mapper.entity_counts['playlists'],
        mapper.entity_counts['tracks'],
        num_negative=len(splits['val_edges']),
        random_seed=args.random_seed
    )

    negative_test = splitter.create_negative_samples(
        splits['test_edges'],
        mapper.entity_counts['playlists'],
        mapper.entity_counts['tracks'],
        num_negative=len(splits['test_edges']),
        random_seed=args.random_seed
    )

    logger.info(f"✅ Negative validation samples: {len(negative_val):,}")
    logger.info(f"✅ Negative test samples: {len(negative_test):,}")

    # 6. Save preprocessed data
    saved_path = save_preprocessed_data(
        output_dir=args.output_dir,
        mappings=mappings,
        edges=edges,
        features=node_features,
        splits=splits,
        negative_val=negative_val,
        negative_test=negative_test,
        entity_counts=mapper.entity_counts,
        reverse_mappings=mapper.reverse_mappings
    )

    # 7. Verify saved data
    verify_preprocessed_data(args.output_dir)

    logger.info("🎉 Preprocessing pipeline completed successfully!")

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📂 Input file: {args.input_file}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🎵 Playlists processed: {len(playlists):,}")
    logger.info(f"🎼 Unique tracks: {mapper.entity_counts['tracks']:,}")
    logger.info(f"🎤 Unique artists: {mapper.entity_counts['artists']:,}")
    logger.info(f"💿 Unique albums: {mapper.entity_counts['albums']:,}")
    logger.info(f"👥 Unique users: {mapper.entity_counts['users']:,}")
    logger.info(f"🔗 Total edges: {sum(len(e) for e in edges.values()):,}")
    logger.info(f"📚 Training edges: {len(splits['train_edges']):,}")
    logger.info(f"🔍 Validation edges: {len(splits['val_edges']):,}")
    logger.info(f"🧪 Test edges: {len(splits['test_edges']):,}")
    logger.info(f"🎲 Random seed used: {args.random_seed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()