# GPT-char

A character-level GPT (Generative Pre-trained Transformer) implementation for text generation and language modeling.

## Overview

GPT-char is a simplified implementation of the GPT architecture that operates at the character level rather than word or token level. This approach allows for fine-grained text generation and is particularly useful for understanding the fundamentals of transformer-based language models.

## Features

- **Character-level tokenization**: Processes text one character at a time
- **Transformer architecture**: Implements multi-head self-attention and feed-forward networks
- **Text generation**: Generate coherent text sequences character by character
- **Training from scratch**: Train on custom datasets
- **Lightweight implementation**: Simplified codebase for educational purposes

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BrijeshSukhadiya/GPT-char.git
cd GPT-char
```

## Quick Start

### Prerequisites

Ensure you have a text file named `text.txt` in the same directory as your script. This file should contain the text data you want to train on.

### Basic Training

```bash
python gpt_char.py
```

### Training Configuration

The model uses the following hyperparameters (defined at the top of the script):

```python
batch_size = 64        # Number of independent sequences processed in parallel
block_size = 256       # Maximum context length for predictions
max_iters = 5000       # Number of training iterations
eval_interval = 500    # How often to evaluate the model
learning_rate = 3e-4   # Learning rate for AdamW optimizer
n_embd = 384          # Embedding dimension
n_head = 6            # Number of attention heads
n_layer = 6           # Number of transformer layers
dropout = 0.2         # Dropout rate
```

### Text Generation

```python
# Generate text
prompt = "The quick brown"
generated_text = model.generate(prompt, max_length=100)
print(generated_text)
```

## Model Architecture

The model consists of:
- **Embedding Layer**: Converts characters to dense vectors
- **Positional Encoding**: Adds position information to embeddings  
- **Transformer Blocks**: Multi-head attention + feed-forward layers
- **Output Layer**: Maps hidden states back to character probabilities

## Dataset Format

The model expects plain text files. Characters are automatically mapped to integers:

```
input.txt
---------
Hello world!
This is a sample text for training.
```

## Acknowledgments
```
- Inspired by Andrej Karpathy's [nanoGPT]
- Based on the "Attention Is All You Need" paper by Vaswani et al.
- Thanks to the open-source community for valuable feedback
```
