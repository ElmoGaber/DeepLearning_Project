# Transformer Self-Attention Visualizer

A minimal, educational implementation of the self-attention mechanism in a Transformer encoder, demonstrated on a medical question-answering example.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [1. Tokenization](#1-tokenization)
  - [2. Embeddings](#2-embeddings)
  - [3. Query, Key, Value Projections](#3-query-key-value-projections)
  - [4. Scaled Dot-Product Attention](#4-scaled-dot-product-attention)
  - [5. Multi-Head Attention (Optional)](#5-multi-head-attention-optional)
- [Example Output](#example-output)
- [Mathematical Formulation](#mathematical-formulation)
- [Extending the Project](#extending-the-project)
- [References](#references)
- [License](#license)

---

## Overview

This project provides a clean, from-scratch NumPy implementation of the **self-attention mechanism** used in Transformer encoders. It processes the medical query:

> *"What are the symptoms of diabetes?"*

and computes attention scores to show how each token relates to every other token in the sequence.

This is designed for **educational purposes** to demystify the inner workings of attention before moving to heavy frameworks like PyTorch or TensorFlow.

---

## Features

- Pure NumPy implementation (no deep learning frameworks required)
- Step-by-step computation of Q, K, V matrices
- Scaled dot-product attention with full mathematical transparency
- Attention weight visualization and interpretation
- Modular code structure ready for extension to Multi-Head Attention
- Clear comments explaining every step of the Transformer encoder logic

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-self-attention.git
cd transformer-self-attention

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.21.0
matplotlib>=3.5.0  # Optional, for visualization
```

---

## Quick Start

Run the main script to see self-attention in action:

```bash
python self_attention.py
```

You should see output showing:
- Tokenized input
- Raw attention scores (QK^T / sqrt(d_k))
- Softmax-normalized attention weights
- Which token each word attends to most strongly

---

## Project Structure

```
transformer-self-attention/
│
├── self_attention.py          # Main implementation
├── multi_head_attention.py    # Extended multi-head version (optional)
├── utils.py                   # Helper functions (softmax, visualization)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── examples/
    └── medical_qa_example.py  # Medical QA context example
```

---

## How It Works

### 1. Tokenization

The input sentence is split into tokens:

```python
sentence = "What are the symptoms of diabetes?"
tokens = sentence.lower().split()
# Result: ['what', 'are', 'the', 'symptoms', 'of', 'diabetes?']
```

In a production Transformer, this would use a subword tokenizer (WordPiece, BPE, or SentencePiece).

### 2. Embeddings

Each token is mapped to a high-dimensional vector. Here we use random embeddings for demonstration:

```python
X = np.random.randn(seq_len, d_model)  # (6, 64)
```

In practice, these are learned embeddings from an embedding layer plus **positional encodings**:

$$PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$$

### 3. Query, Key, Value Projections

The core of attention: each token creates three representations:

```python
Q = X @ W_q   # What am I looking for?  (Queries)
K = X @ W_k   # What do I contain?      (Keys)
V = X @ W_v   # What information do I have? (Values)
```

These are learned linear projections that allow the model to focus on different aspects of meaning.

### 4. Scaled Dot-Product Attention

The attention mechanism computes compatibility between every Query and every Key:

```python
scores = (Q @ K.T) / np.sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V
```

**Why scale by sqrt(d_k)?**  
As $d_k$ grows, the dot products become larger, pushing the softmax function into regions with extremely small gradients. Scaling stabilizes training.

### 5. Multi-Head Attention (Optional)

The full Transformer uses multiple attention heads in parallel:

```python
# h heads, each with dimension d_k = d_model / h
head_1 = attention(Q_1, K_1, V_1)
head_2 = attention(Q_2, K_2, V_2)
...
head_h = attention(Q_h, K_h, V_h)

# Concatenate and project
output = concat(head_1, ..., head_h) @ W_o
```

Each head learns different types of relationships (syntax, semantics, long-range dependencies).

---

## Example Output

```
Tokens: ['what', 'are', 'the', 'symptoms', 'of', 'diabetes?']

--- Attention Scores (QK^T / sqrt(d_k)) ---
[[ 0.234 -0.156  0.089 ... ]
 [ ... ]]

--- Attention Weights (Softmax) ---
[[0.142 0.127 0.128 0.132 0.129 0.142]
 [0.129 0.146 0.128 0.133 0.129 0.135]
 ...]

--- Dominant Attention per Token ---
'what' attends most to 'diabetes?' (weight: 0.142)
'are' attends most to 'symptoms' (weight: 0.146)
'symptoms' attends most to 'diabetes?' (weight: 0.138)
'of' attends most to 'symptoms' (weight: 0.144)
'diabetes?' attends most to 'what' (weight: 0.142)

Output shape: (6, 8)
```

Notice how **"symptoms"** and **"diabetes?"** attend to each other, reflecting their semantic relationship in the medical query.

---

## Mathematical Formulation

The self-attention operation is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$ (Queries)
- $K \in \mathbb{R}^{n \times d_k}$ (Keys)
- $V \in \mathbb{R}^{n \times d_k}$ (Values)
- $n$ = sequence length
- $d_k$ = key dimension

For a full encoder layer:

$$\text{EncoderOutput} = \text{LayerNorm}(X + \text{Attention}(X))$$
$$\text{FFNOutput} = \text{LayerNorm}(\text{EncoderOutput} + \text{FFN}(\text{EncoderOutput}))$$

Where FFN is a feed-forward network: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$

---

## Extending the Project

### Add Positional Encodings

```python
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

X = embeddings + positional_encoding(seq_len, d_model)
```

### Add Multi-Head Attention

See `multi_head_attention.py` for a complete implementation that splits Q, K, V into $h$ heads, computes attention in parallel, and concatenates the results.

### Add Masking (for Decoder)

For decoder self-attention, prevent positions from attending to future tokens:

```python
def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    return mask * -1e9  # Add to scores before softmax
```

### Add Feed-Forward Network

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1) @ self.W2  # ReLU activation
```

### Add Layer Normalization

```python
def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
```

---

## References

1. **Vaswani et al. (2017)** - "Attention Is All You Need"  
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Devlin et al. (2018)** - "BERT: Pre-training of Deep Bidirectional Transformers"  
   [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

3. **The Illustrated Transformer** by Jay Alammar  
   [Blog Post](https://jalammar.github.io/illustrated-transformer/)

4. **Hugging Face Transformers Documentation**  
   [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Areas for improvement:
- Add visualization of attention heatmaps using Matplotlib/Seaborn
- Implement full encoder and decoder stacks
- Add training loop with backpropagation
- Benchmark against PyTorch's MultiHeadAttention

Please open an issue or submit a pull request.

---
