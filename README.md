# Bangla Audio Analysis: Vector Embeddings and Cosine Similarity

## Quick Access Links

- **📓 Google Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/17OWIyp0sD7Sxmxur6e6hnM-5aZVTgV09?usp=sharing)
- **📊 Dataset**: [Download from Google Drive](https://drive.google.com/drive/folders/1RYNeUIz5neP0ojlBK04nJmqL4XggaR2o?usp=sharing)

## Overview

This project implements a comprehensive pipeline for analyzing Bangla (Bengali) audio recordings using deep learning-based vector embeddings to measure speaker similarity through cosine similarity analysis. The system utilizes Facebook's wav2vec2-large-xlsr-53 multilingual model to extract high-dimensional audio representations and performs detailed similarity analysis for speaker identification and verification tasks.

## Features

- **Audio Preprocessing**: Automated pipeline for loading, normalizing, and standardizing audio files
- **Deep Learning Embeddings**: Uses wav2vec2-large-xlsr-53 for 1024-dimensional audio representations
- **Similarity Analysis**: Comprehensive cosine similarity and distance calculations
- **Vector Database**: FAISS-based efficient similarity search system
- **Advanced Analytics**: Clustering, dimensionality reduction, and performance evaluation
- **Visualization Suite**: Multiple graph types for data exploration and results analysis
- **Speaker Identification**: Automated speaker labeling and verification capabilities

## Dataset

- **Total Files**: 14 Bangla audio recordings (.m4a format)
- **Speakers**: 2 distinct speakers (Person_1: 10 files, Person_2: 4 files)
- **Duration Range**: 4.03 - 30.46 seconds (mean: 20.89s)
- **Sample Rate**: 16kHz (standardized)
- **File Size**: 0.05 - 0.35 MB per file

## Installation

### Prerequisites

```bash
Python 3.7+
CUDA-compatible GPU (optional, for faster processing)
```

### Required Packages

```bash
pip install transformers torch torchaudio librosa soundfile numpy pandas scikit-learn faiss-cpu seaborn matplotlib
```

### For GPU acceleration (optional):
```bash
pip install faiss-gpu
```

## Usage

### Option 1: Google Colab (Recommended)

The easiest way to run this project is using our pre-configured Google Colab notebook:

1. **Open the notebook**: [Bangla Audio Analysis Colab](https://colab.research.google.com/drive/17OWIyp0sD7Sxmxur6e6hnM-5aZVTgV09?usp=sharing)
2. **Download the dataset**: [Audio files from Google Drive](https://drive.google.com/drive/folders/1RYNeUIz5neP0ojlBK04nJmqL4XggaR2o?usp=sharing)
3. **Upload dataset** to your Google Drive or Colab environment
4. **Run all cells** in sequence (Cells 1-10)

### Option 2: Local Setup

#### Quick Start

```python
# Import necessary libraries
from bangla_audio_processor import BanglaAudioProcessor, VectorDatabase, SimilarityAnalyzer

# Initialize processor
processor = BanglaAudioProcessor()

# Process audio files
audio_folder = "path/to/your/audio/files"
results = processor.process_dataset(audio_folder)

# Create vector database
vector_db = VectorDatabase(embedding_dim=1024)
vector_db.add_vectors(results['embeddings'], metadata)

# Perform similarity search
query_embedding = results['embeddings'][0]
similar_results = vector_db.search_similar(query_embedding, k=3)
```

### Complete Analysis Pipeline

Run the provided Jupyter notebook cells in sequence:

1. **Cell 1**: Install required packages
2. **Cell 2**: Import libraries and mount data source
3. **Cell 3**: Load and explore audio dataset
4. **Cell 4**: Define audio processor class
5. **Cell 5**: Extract embeddings from all audio files
6. **Cell 6**: Calculate similarity matrices and distances
7. **Cell 7**: Generate comprehensive visualizations
8. **Cell 8**: Perform advanced clustering analysis
9. **Cell 9**: Create vector database and similarity search
10. **Cell 10**: Generate summary and save results

## Technical Architecture

### Audio Processing Pipeline

1. **Loading**: Audio files loaded using librosa with 16kHz resampling
2. **Normalization**: Amplitude normalization for consistent signal levels
3. **Trimming**: Automatic silence removal (20dB threshold)
4. **Standardization**: Fixed 10-second duration through padding/truncation
5. **Feature Extraction**: wav2vec2 model generates 1024-dim embeddings

### Model Details

- **Base Model**: facebook/wav2vec2-large-xlsr-53
- **Architecture**: Transformer-based self-supervised learning
- **Language Support**: 53 languages including Bengali
- **Output Dimension**: 1024 features per audio segment
- **Processing**: Mean pooling of temporal features

### Similarity Metrics

- **Cosine Similarity**: Angular similarity between embedding vectors (-1 to 1)
- **Cosine Distance**: Dissimilarity measure (1 - cosine similarity)
- **Vector Database**: FAISS IndexFlatIP for efficient similarity search

## Results Summary

### Key Performance Metrics

- **Processing Success Rate**: 100% (14/14 files)
- **Embedding Dimension**: 1024 features
- **Total Similarity Pairs**: 91 comparisons
- **Clustering Accuracy**: 71.43%
- **Rank-1 Search Accuracy**: 78.57%
- **Rank-3 Search Accuracy**: 85.71%

### Similarity Analysis

- **Same-Speaker Similarity**: 0.9993 (average)
- **Cross-Speaker Similarity**: 0.9996 (average)
- **Separation Margin**: -0.0003 (indicates poor discrimination)
- **Score Range**: 0.9988 - 1.0000 (very compressed)

### Key Findings

- **Limited Speaker Discrimination**: The wav2vec2 model shows poor speaker separation
- **Language-Focused Embeddings**: Model prioritizes linguistic over speaker-specific features
- **High Similarity Scores**: Most comparisons yield >99.9% similarity regardless of speaker
- **Clustering Challenges**: Moderate accuracy suggests overlapping speaker characteristics

## Visualizations

The project generates multiple visualization types:

1. **Similarity Heatmaps**: Cosine similarity and distance matrices
2. **Distribution Analysis**: Histogram comparisons of same vs. different speakers
3. **Statistical Plots**: Box plots and accuracy metrics
4. **Clustering Visualization**: t-SNE plots with true and predicted labels
5. **Search Performance**: Ranking accuracy and per-file success rates

## File Structure

```
bangla-audio-analysis/
├── notebooks/
│   ├── bangla_audio_analysis.ipynb
│   └── data_exploration.ipynb
├── src/
│   ├── audio_processor.py
│   ├── vector_database.py
│   └── similarity_analyzer.py
├── data/
│   ├── audio_files/
│   ├── embeddings/
│   └── results/
├── results/
│   ├── similarity_matrix.npy
│   ├── distance_matrix.npy
│   ├── audio_embeddings.npy
│   ├── audio_metadata.csv
│   └── similarity_analysis.csv
├── visualizations/
│   ├── similarity_heatmap.png
│   ├── clustering_analysis.png
│   └── search_performance.png
└── README.md
```

## Limitations and Future Work

### Current Limitations

- **Poor Speaker Separation**: wav2vec2 not optimized for speaker identification
- **Compressed Similarity Space**: Insufficient variation for reliable discrimination
- **Limited Dataset Size**: Small sample size may affect generalization
- **Imbalanced Data**: Unequal speaker representation (10 vs 4 files)

### Proposed Improvements

1. **Speaker-Specific Fine-tuning**: Adapt model for speaker identification tasks
2. **Alternative Models**: Experiment with speaker embedding models (x-vectors, d-vectors)
3. **Data Augmentation**: Expand dataset with more speakers and longer recordings
4. **Multi-modal Features**: Combine acoustic with prosodic and linguistic features
5. **Advanced Clustering**: Implement hierarchical clustering and ensemble methods

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis feature'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Dependencies

- **transformers**: Hugging Face transformers library
- **torch**: PyTorch deep learning framework
- **torchaudio**: Audio processing for PyTorch
- **librosa**: Audio analysis library
- **soundfile**: Audio I/O operations
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities
- **faiss**: Facebook AI Similarity Search
- **seaborn**: Statistical data visualization
- **matplotlib**: Plotting library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bangla_audio_analysis,
  title={Bangla Audio Analysis: Vector Embeddings and Cosine Similarity},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[mansu123]/bangla-audio-analysis}
}
```

## Contact

- **Author**: [Mansuba ]
- **Email**: [mansubatabassum9@gmail.com]
- **GitHub**: [mansu123]
- **Project Link**: [https://github.com/mansu123/bangla-audio-analysis]

## Acknowledgments

- Facebook AI Research for the wav2vec2-large-xlsr-53 model
- Hugging Face for the transformers library
- The open-source community for the various libraries used in this project
