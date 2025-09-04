# GNN Music Recommender System

**A Extended Research Project Report submitted to the University of Manchester for the degree of MSc Data Science in the Faculty of Humanities**

## Project Overview

This project investigates the effectiveness of Graph Neural Networks (GNNs) for music recommendation systems, specifically comparing bipartite and heterogeneous graph structures and evaluating the contribution of multi-modal features versus structure-only approaches. The research provides empirical evidence on optimal graph construction methodologies for music recommendation scenarios.

### Research Objectives

1. **Graph Architecture Comparison**: Evaluate performance differences between bipartite user-item graphs and heterogeneous multi-node-type structures
2. **Feature Contribution Analysis**: Assess the value-added of multi-modal node features (playlist metadata, track characteristics, user behaviour) against pure collaborative filtering signals
3. **Implementation Guidelines**: Develop a practical configuration selection methodology for music recommendation systems

## Dataset

This project uses a carefully curated subset of the **Spotify Million Playlist Dataset Challenge**, processed and scaled for research purposes.

### Original Dataset Source

- **Source**: [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- **Original Scale**: 1 million playlists, 2+ million unique tracks, 66+ million playlist-track interactions
- **Format**: JSON files containing user-generated playlists with metadata and track information
- **Time Period**: Playlists created between January 2010 and October 2017

### Processed Dataset

**File**: `data/processed/spotify_scaled_hybrid_tiny.json`
This file represents a strategically sampled and preprocessed subset of the original dataset

## Installation and Setup

1. **Clone the repository**
```bash
git clone https://github.com/Serena1014/gnn-music-recommender.git
cd gnn-music-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Experimental Pipeline

### Phase 1: Data Exploration

**Notebook**: `notebooks/01_data_exploration.ipynb`

This notebook covers:
- Dataset loading and initial inspection
- Statistical analysis of user-playlist-track relationships
- Data distribution visualisation

**Key Outputs**:
- Dataset statistics and summaries
- Visualisation of data distributions


### Phase 2: Sampling and Quality Assessment

**Notebooks**: 
- `notebooks/02_hybrid_sampling_analysis.ipynb` - Sampling methodology evaluation
- `notebooks/03_data_quality_assessment.ipynb` - Data validation

These notebooks implement:
- Hybrid sampling strategies for graph construction
- Data quality metrics and validation procedures

**Key Outputs**:
- Optimal sampling configuration
- Data quality reports
- Graph structure statistics

### Phase 3: Full Experimental Pipeline

**Notebook**: `notebooks/04_full_experiment.ipynb`

This comprehensive notebook executes:
- Complete model training pipeline
- Comparative analysis of graph architectures
- Feature contribution evaluation
- Statistical significance testing
- Results visualisation

## Reproducibility Instructions

### Complete Experimental Reproduction

1. **Environment Setup**
```bash
pip install -r requirements.txt
```

2. **Data Exploration** (Optional - for understanding)
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

3. **Run Complete Experiments**
```bash
jupyter notebook notebooks/04_full_experiment.ipynb
```

This notebook will:
- Load the preprocessed dataset
- Execute experimental phases
- Generate all results files in the `results/` directory
- Perform statistical analysis and visualisation

## Results Structure

### Experimental Outputs

Each experimental phase generates:

1. **Complete Results** (`complete_results_*.json`):
2. **Experiment Summaries** (`experiment_summary_*.json`):
3. **Model Checkpoints** (`models/` directories):
4. **Training Logs** (`logs/` directories):

### Performance Metrics

The system evaluates models using:
- **Recall@K** (K = 10, 20, 50): Fraction of relevant items retrieved
- **NDCG@K** (K = 10, 20, 50): Normalised Discounted Cumulative Gain
- **Training Metrics**: Loss convergence, training time, memory usage

## Statistical Analysis

### Methodology

All experiments follow rigorous statistical procedures:

1. **Multiple Random Seeds**: Each configuration tested with ≥5 different seeds
2. **Statistical Testing**: Paired t-tests for performance comparisons
3. **Confidence Intervals**: 95% confidence intervals for all reported metrics
4. **Effect Size**: Cohen's d for practical significance assessment

### Significance Testing

Results include:
- P-values for architecture comparisons
- Effect size calculations
- Confidence interval reporting
- Multiple comparison corrections (Bonferroni)
