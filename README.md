# LyricGen - AI-Powered Lyric Completion Tool

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/emanafi/lyricgen)

## Overview

LyricGen is a sophisticated machine learning-based lyric generation tool that leverages a custom Transformer-based neural network to generate and complete song lyrics. The model processes multilingual datasets from Genius and generates contextually coherent lyrical content across three languages: English, French, and Arabic.

## Features

- **Multilingual Support**: Generates lyrics in English, French, and Arabic with a shared tokenizer for consistent token IDs
- **Advanced Data Preprocessing**: Comprehensive text cleaning, filtering, and normalization
- **Transformer Architecture**: Custom implementation with multi-head attention, positional encoding, and layer normalization
- **Dual Decoding Modes**: `strict` for controlled deterministic behavior and `quality` for fluent generation with anti-repetition controls
- **Richer Evaluation**: Exact Match, BLEU, Top-1/Top-3/Top-5 next-token accuracy, and perplexity
- **Weight Tying**: Shared embedding/output weights for parameter efficiency and regularization
- **LR Warmup**: Linear warmup with inverse-sqrt decay for stable Transformer training
- **Efficient Training**: Optimized for Kaggle environment with ~70K training samples

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

- **Vocabulary Size**: Shared tokenizer effective size (capped by 15,000 + padding)
- **Sequence Length**: 80 tokens for longer lyric context
- **Embedding Dimension**: 192
- **Attention Heads**: 6 (key_dim=32 per head)
- **Feed-Forward Dimension**: 768
- **Decoder Layers**: 3
- **Total Parameters**: Approximately 4.2M (right-sized for ~70K samples)
- **Key Components**:
  - Positional encoding with sqrt(d_model) scaling for sequence awareness
  - Multi-head self-attention layers
  - Layer normalization and dropout for regularization
  - Weight tying between embedding and output layers
  - Shared tokenizer across all supported languages

## Usage

### Training the Model

1. Load and preprocess the Genius dataset
2. Configure one shared tokenizer across English, French, and Arabic
3. Train the Transformer model with padding-aware sample weights and label smoothing
4. Evaluate with Top-1/Top-3/Top-5 next-token accuracy and perplexity

### Generating Lyrics

Use the interactive prediction function:

```python
predict_next_lyrics(
    seed_text="your starting lyrics here",
    language='en',  # 'en', 'fr', or 'ar'
    num_words=8,
  mode='quality'   # 'quality' or 'strict'
)
```

**Decoding Modes**:

- **strict**: Deterministic and conservative, better for reproducible comparison runs
- **quality**: Nucleus sampling + anti-repetition controls, better fluency for sentence generation

## Preprocessing Pipeline

1. **Language Filtering**: Select English, French, and Arabic lyrics
2. **Text Cleaning**:
   - English & French: Lowercase conversion, punctuation removal
   - Arabic: Preserve Unicode characters and original case
3. **Special Token Addition**: Add `<sos>` (start) and `<eos>` (end) markers
4. **Tokenization**: Shared multilingual tokenizer with `<OOV>` handling
5. **Sequence Padding**: Post-padding to 80-token length (real tokens first, causal mask prevents attending to trailing padding)
6. **Dataset Splitting**: 70% training, 15% validation, 15% test

## Model Training Details

- **Optimizer**: Adam with LR warmup schedule (peak 5e-4, 2000 warmup steps, inverse-sqrt decay)
- **Loss Function**: Sparse categorical crossentropy with `from_logits=True` and label smoothing (`0.1`)
- **Training Strategy**: Autoregressive next-token prediction
- **Regularization**: Dropout, layer normalization, label smoothing, and weight tying
- **Padding Handling**: Sample weights exclude padded positions from optimization and metrics

## Evaluation Metrics

The model's performance is evaluated using:

- **Exact Match**: Strict token-position overlap against references
- **BLEU Scores**: N-gram overlap between generated and reference continuations
- **Top-1 / Top-3 / Top-5 Accuracy**: Masked next-token prediction quality
- **Perplexity**: `exp(loss)` as a language-model confidence/fit indicator
- **Strict vs Quality Comparison**: Side-by-side evaluation across decoding presets

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
