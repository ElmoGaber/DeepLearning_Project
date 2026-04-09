import numpy as np

np.random.seed(42)

# ── helpers ──────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def one_hot(char, vocab):
    v = np.zeros(len(vocab))
    v[vocab[char]] = 1.0
    return v

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def show(name, val):
    print(f"  {name}:  {np.round(val, 4)}")


# ── config ───────────────────────────────────────────────────
vocab    = {'d': 0, 'o': 1, 'g': 2, 's': 3}
k        = len(vocab)   # input / output size = 4
d        = 3            # hidden nodes
lr       = 0.1          # learning rate
epochs   = 15

sequence = ['d', 'o', 'g']
target   = 's'
X_seq    = [one_hot(c, vocab) for c in sequence]
y_true   = one_hot(target, vocab)


# ════════════════════════════════════════════════════════════
# STEP 1 — Initialize weights randomly
# ════════════════════════════════════════════════════════════
W_H = np.array([[0.1, 0.5, 0.1],
                [0.5, 0.9, 0.3],
                [0.3, 0.2, 0.1]])   # (d×d)

W_X = np.array([[0.6, 0.8, 0.4, 0.8],
                [0.2, 0.2, 0.8, 0.7],
                [0.9, 0.8, 0.1, 0.2]])  # (d×k)

W_Y = np.array([[0.9, 0.8, 0.3],
                [0.2, 0.3, 0.4],
                [0.6, 0.9, 0.1],
                [0.5, 0.0, 0.3]])   # (k×d)

print("=" * 55)
print("  RNN — Backpropagation Through Time (BPTT)")
print("=" * 55)
print(f"\n  vocab={list(vocab.keys())}  |  k={k}  |  d={d}  |  lr={lr}")
print(f"  sequence: {sequence}  →  target: '{target}'\n")


# ════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════
for epoch in range(1, epochs + 1):

    # ── STEP 2 — Forward propagation ────────────────────────
    h_prev = np.zeros(d)
    hs = {0: h_prev}   # store all hidden states for BPTT
    as_ = {}           # store pre-activation values

    for t, X in enumerate(X_seq, 1):
        a       = W_H @ hs[t-1] + W_X @ X
        hs[t]   = np.tanh(a)
        as_[t]  = a

    y_pred = softmax(W_Y @ hs[len(X_seq)])

    # ── STEP 3 — Compute loss (cross-entropy) ───────────────
    loss = cross_entropy(y_pred, y_true)

    # ── STEP 4 — Backpropagation through time ───────────────
    # Gradient of loss w.r.t. output
    dy = y_pred - y_true                            # (k,)

    # dL/dW_Y  =  dy · h_T^T
    dW_Y = np.outer(dy, hs[len(X_seq)])            # (k×d)

    # Gradient flowing back into hidden layer at t=T
    dh_next = W_Y.T @ dy                           # (d,)

    dW_H = np.zeros_like(W_H)
    dW_X = np.zeros_like(W_X)

    # ── Unroll backwards through time ───────────────────────
    T = len(X_seq)
    for t in range(T, 0, -1):
        # tanh derivative:  1 - h²
        dtanh = (1 - hs[t] ** 2)                   # (d,)

        # delta = gradient reaching this time step
        delta = dh_next * dtanh                     # (d,)

        # Accumulate weight gradients
        dW_H += np.outer(delta, hs[t-1])           # (d×d)
        dW_X += np.outer(delta, X_seq[t-1])        # (d×k)

        # Pass gradient further back in time
        dh_next = W_H.T @ delta                    # (d,)

    # ── STEP 5 — Update weights ─────────────────────────────
    W_Y -= lr * dW_Y
    W_H -= lr * dW_H
    W_X -= lr * dW_X

    # ── Log ─────────────────────────────────────────────────
    pred_char = list(vocab)[np.argmax(y_pred)]
    if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
        print(f"  Epoch {epoch:02d}  |  Loss = {loss:.4f}  |  "
              f"y_pred = {np.round(y_pred, 3)}  |  predicted = '{pred_char}'")

        if epoch == 1:
            print()
            print("  --- Gradients at epoch 1 ---")
            show("dy  (dL/dy_pred)", dy)
            show("dW_Y (sample row 0)", dW_Y[0])
            show("dW_H (sample row 0)", dW_H[0])
            show("dW_X (sample row 0)", dW_X[0])
            print()


# ── Final result ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULT")
print("=" * 55)
y_final   = softmax(W_Y @ np.tanh(W_H @ np.zeros(d) +
            W_X @ one_hot('d', vocab)))

h = np.zeros(d)
for X in X_seq:
    h = np.tanh(W_H @ h + W_X @ X)
y_final = softmax(W_Y @ h)

pred = list(vocab)[np.argmax(y_final)]
print(f"\n  After {epochs} epochs:")
print(f"  y_pred = {np.round(y_final, 4)}")
print(f"  predicted = '{pred}'  |  target = '{target}'")
print(f"  Correct? {'✓' if pred == target else '✗'}\n")

print("  Updated weight matrices:")
show("W_Y[0]", W_Y[0])
show("W_H[0]", W_H[0])
show("W_X[0]", W_X[0])
