# GPT Text Generator for Eminem Tracks

This project is a custom implementation of a GPT-based language model trained on all tracks by Eminem. It uses a transformer architecture to generate new text in the style of Eminem's lyrics.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Generating Text](#generating-text)
- [Usage](#usage)
- [References](#references)

## Overview

This project implements a GPT-like model from scratch using PyTorch. The model is trained to predict the next character in a sequence of text, which allows it to generate new sequences of text based on a given prompt. The model has been specifically trained on the lyrics of Eminem's songs.

## Model Architecture

The model consists of several layers of self-attention and feedforward neural networks, following the Transformer architecture. Key components include:

- **Multi-Head Self-Attention**: The core mechanism that allows the model to attend to different parts of the input sequence.
- **Feedforward Neural Network**: A simple MLP applied after the self-attention mechanism.
- **Positional Embeddings**: Since this is a sequence model, we use positional embeddings to give the model a sense of order in the input sequence.
- **Layer Normalization**: Applied between layers to stabilize and speed up training.

### Hyperparameters

- **Batch Size**: 64
- **Block Size**: 256 (maximum length of context for predictions)
- **Embedding Size**: 384
- **Number of Heads**: 6
- **Number of Layers**: 6
- **Dropout**: 0.2
- **Learning Rate**: 3e-4
- **Max Iterations**: 6

## Requirements

To run this project, you will need the following libraries:

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support for GPU training)
- tqdm
- numpy

You can install dependencies via `pip`:

```bash
pip install torch tqdm numpy
```
## Dataset
The dataset consists of all lyrics from Eminem's tracks, stored in a text file all_tracks.txt. Make sure this file is present in the project's root directory before training the model.


## Generating Text
Once the model has been trained, you can generate new text using the following method in the script:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist()
print(generated_text)
```
This will generate 500 new tokens (characters) based on the model's training.

You can also write the generated text to a file using the following line:

```python
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
This will save 10,000 characters of generated text to more.txt.
```
## Usage
Clone the repository:

```bash
git clone https://github.com/githumster/Simple_GPT.git
```
Place the text file all_tracks.txt in the root directory.

## References
This project is inspired by the following resource:

[Attention Is All You Need (Transformer paper)](https://arxiv.org/pdf/1706.03762)
