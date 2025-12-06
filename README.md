# LyricGen - AI-Powered Lyric Completion Tool

## Overview

LyricGen is a sophisticated machine learning-based lyric generation tool that leverages a custom Transformer-based neural network to generate and complete song lyrics. The model processes multilingual datasets from Genius and generates contextually coherent lyrical content across three languages: English, French, and Arabic.

## Features

- **Multilingual Support**: Generates lyrics in English, French, and Arabic with language-specific tokenizers
- **Advanced Data Preprocessing**: Comprehensive text cleaning, filtering, and normalization
- **Transformer Architecture**: Custom implementation with multi-head attention, positional encoding, and layer normalization
- **Interactive Generation**: User-friendly prediction interface with customizable parameters
- **Temperature Control**: Adjustable creativity levels (0.3-1.2) for diverse output styles
- **Performance Evaluation**: BLEU score assessment for model quality
- **Efficient Training**: Optimized for Kaggle environment with ~27K training samples

## Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn nltk tensorflow matplotlib
```

## Dataset

The model uses the **Genius Song Lyrics with Language Information** dataset from Kaggle, containing song lyrics with metadata. The dataset undergoes extensive preprocessing including:

- Filtering for English, French, and Arabic lyrics only
- Removal of special characters, HTML tags, and structural markers
- Deduplication of redundant lyrics
- Language-specific text normalization

## Model Architecture

The model implements a custom Transformer-based architecture:

- **Vocabulary Size**: 15,000 words per language for comprehensive coverage
- **Sequence Length**: 50 tokens for optimal context window
- **Embedding Dimension**: 256
- **Attention Heads**: 8 multi-head attention mechanisms
- **Feed-Forward Dimension**: 512
- **Total Parameters**: Approximately 10-12M parameters
- **Key Components**:
  - Positional encoding for sequence awareness
  - Multi-head self-attention layers
  - Layer normalization and dropout for regularization
  - Language-specific tokenization

## Usage

### Training the Model

1. Load and preprocess the Genius dataset
2. Configure language-specific tokenizers (English, French, Arabic)
3. Train the Transformer model with the prepared sequences
4. Evaluate performance using BLEU scores

### Generating Lyrics

Use the interactive prediction function:

```python
predict_next_lyrics(
    seed_text="your starting lyrics here",
    language='en',  # 'en', 'fr', or 'ar'
    num_words=8,
    temperature=0.7  # 0.3-1.2 for creativity control
)
```

**Temperature Guidelines**:

- **0.3-0.5**: Conservative, predictable outputs
- **0.6-0.8**: Balanced mode (recommended)
- **0.9-1.2**: Creative, experimental outputs

## Preprocessing Pipeline

1. **Language Filtering**: Select English, French, and Arabic lyrics
2. **Text Cleaning**:
   - English & French: Lowercase conversion, punctuation removal
   - Arabic: Preserve Unicode characters and original case
3. **Special Token Addition**: Add `<sos>` (start) and `<eos>` (end) markers
4. **Tokenization**: Language-specific vocabulary building with `<OOV>` handling
5. **Sequence Padding**: Normalize to 50-token length
6. **Dataset Splitting**: 70% training, 15% validation, 15% test

## Model Training Details

- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Sparse categorical crossentropy
- **Training Strategy**: Autoregressive next-token prediction
- **Data Augmentation**: Language-aware processing
- **Regularization**: Dropout and layer normalization

## Evaluation Metrics

The model's performance is evaluated using:

- **BLEU Scores**: Measure similarity between generated and reference lyrics
- **Perplexity**: Assess model confidence
- **Language-Specific Metrics**: Per-language performance analysis

## Key Highlights

- Handles right-to-left text (Arabic) and left-to-right languages (English, French)
- Maintains accented characters for French
- Optimized for computational efficiency on Kaggle
- Interactive interface for creative lyric exploration
- Suitable for songwriting assistance, creative writing, and language learning

## Applications

- **Songwriting Assistance**: Generate continuation ideas for lyrics in progress
- **Creative Writing**: Explore different stylistic directions
- **Language Learning**: Study natural language patterns across languages
- **Educational Demonstrations**: Showcase modern NLP text generation capabilities

## Author

**Eman Sarah Afi**  
Fall 2024

---

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/emanafi/lyricgen)
