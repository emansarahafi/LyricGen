# LyricGen - AI-Powered Lyric Completion Tool

## Overview
LyricGen is a machine learning-based lyric generation tool that leverages a Transformer-based neural network to generate and complete song lyrics. It processes multilingual datasets and generates coherent lyrical content.

## Features
- Supports English, French, and Arabic lyrics
- Data preprocessing including text cleaning and filtering
- Uses a Transformer-based deep learning model for lyric generation
- BLEU score evaluation for model performance assessment

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn nltk tensorflow matplotlib
```

## Dataset
The model uses a dataset containing song lyrics with language information. The dataset is preprocessed to remove special characters, redundant lyrics, and structural tags.

## Usage
1. Load and preprocess the dataset.
2. Train the model using the provided TensorFlow-based Transformer architecture.
3. Evaluate the model using BLEU scores.
4. Generate lyrics based on a given prompt.

## Preprocessing Steps
- Filter dataset to include English, French, and Arabic lyrics.
- Convert text to lowercase (except Arabic).
- Remove punctuation and special characters.
- Tokenize and pad sequences for training.

## Model Architecture
- Uses TensorFlow and Keras for model development.
- Implements Multi-Head Attention layers.
- Includes embedding, dense layers, and layer normalization.

## Evaluation
- The model's performance is evaluated using BLEU scores, which measure the similarity between generated lyrics and real lyrics.

## Author
Eman Sarah Afi (Fall 2024)

