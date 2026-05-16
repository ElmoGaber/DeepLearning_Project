import numpy as np

# -------------------------------------------------------------------------
# 1. Setup: Tokenization and Hyperparameters
# -------------------------------------------------------------------------
sentence = "What are the symptoms of diabetes?"
tokens = sentence.lower().split()  # Simple word-level tokenization

seq_len = len(tokens)
d_model = 64   # Model embedding dimension
d_k = 8        # Key/Query/Value dimension for a single head

np.random.seed(42)

# -------------------------------------------------------------------------
# 2. Simulate Input Embeddings
# -------------------------------------------------------------------------
# In a real Transformer, these are learned lookup embeddings + positional encodings.
X = np.random.randn(seq_len, d_model)  # Shape: (seq_len, d_model)

# -------------------------------------------------------------------------
# 3. Learned Projection Matrices
# -------------------------------------------------------------------------
# W_q, W_k, W_v are learned parameters of shape (d_model, d_k)
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)

# -------------------------------------------------------------------------
# 4. Compute Q, K, V
# -------------------------------------------------------------------------
Q = X @ W_q   # Queries: (seq_len, d_k)
K = X @ W_k   # Keys:    (seq_len, d_k)
V = X @ W_v   # Values:  (seq_len, d_k)

# -------------------------------------------------------------------------
# 5. Compute Attention Scores and Weights
# -------------------------------------------------------------------------
# Raw attention scores (scaled dot-product)
scores = (Q @ K.T) / np.sqrt(d_k)   # Shape: (seq_len, seq_len)

# Softmax over the last dimension (keys)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)  # Shape: (seq_len, seq_len)

# -------------------------------------------------------------------------
# 6. Compute Self-Attention Output
# -------------------------------------------------------------------------
output = attention_weights @ V       # Shape: (seq_len, d_k)

# -------------------------------------------------------------------------
# 7. Display Results
# -------------------------------------------------------------------------
print("Tokens:", tokens)
print("\n--- Attention Scores (QK^T / sqrt(d_k)) ---")
print(np.round(scores, 3))

print("\n--- Attention Weights (Softmax) ---")
print(np.round(attention_weights, 3))

# Show which tokens each position attends to most strongly
print("\n--- Dominant Attention per Token ---")
for i, token in enumerate(tokens):
    attn = attention_weights[i]
    dominant_idx = np.argmax(attn)
    print(f"'{token}' attends most to '{tokens[dominant_idx]}' (weight: {attn[dominant_idx]:.3f})")

print(f"\nOutput shape: {output.shape}")