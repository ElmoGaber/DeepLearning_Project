import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def one_hot(char, vocab):
    v = np.zeros(len(vocab))
    v[vocab[char]] = 1.0
    return v

def show(name, val):
    print(f"  {name}:")
    if val.ndim == 1:
        print("   ", np.round(val, 4))
    else:
        for row in val:
            print("   ", np.round(row, 4))


# ══════════════════════════════════════════════
# PART 1 — Vanilla RNN
# ══════════════════════════════════════════════
print("\n" + "="*50)
print("  PART 1 — Vanilla RNN  (word = 'dogs')")
print("="*50)

vocab    = {'d': 0, 'o': 1, 'g': 2, 's': 3}
k, d     = 4, 3
sequence = ['d', 'o', 'g']
target   = 's'

W_H = np.array([[0.1, 0.5, 0.1],
                [0.5, 0.9, 0.3],
                [0.3, 0.2, 0.1]])

W_X = np.array([[0.6, 0.8, 0.4, 0.8],
                [0.2, 0.2, 0.8, 0.7],
                [0.9, 0.8, 0.1, 0.2]])

W_Y = np.array([[0.9, 0.8, 0.3],
                [0.2, 0.3, 0.4],
                [0.6, 0.9, 0.1],
                [0.5, 0.0, 0.3]])

h = np.zeros(d)

for t, char in enumerate(sequence, 1):
    print(f"\n--- t={t}  input='{char}' ---")
    X = one_hot(char, vocab)
    a = W_H @ h + W_X @ X
    h = tanh(a)
    y = softmax(W_Y @ h)
    show("a_t", a)
    show("h_t", h)
    show("y_t", y)
    print(f"  predicted: '{list(vocab)[np.argmax(y)]}'")

y_true = one_hot(target, vocab)
E = np.sum((y - y_true) ** 2)
print(f"\n  Squared Error = {round(E, 6)}")


# ══════════════════════════════════════════════
# PART 2 — LSTM Forward Pass
# ══════════════════════════════════════════════
print("\n" + "="*50)
print("  PART 2 — LSTM Forward Pass")
print("="*50)

input_size  = 3
hidden_size = 4

def W(r, c): return np.round(np.random.uniform(-0.5, 0.5, (r, c)), 2)
def b(n):    return np.zeros(n)

W_xf, W_hf, b_f = W(hidden_size, input_size), W(hidden_size, hidden_size), b(hidden_size)
W_xi, W_hi, b_i = W(hidden_size, input_size), W(hidden_size, hidden_size), b(hidden_size)
W_xc, W_hc, b_c = W(hidden_size, input_size), W(hidden_size, hidden_size), b(hidden_size)
W_xo, W_ho, b_o = W(hidden_size, input_size), W(hidden_size, hidden_size), b(hidden_size)

X_seq = [
    np.array([1.0, 0.0, 0.5]),
    np.array([0.0, 1.0, 0.3]),
    np.array([0.5, 0.5, 1.0]),
]

H = np.zeros(hidden_size)
C = np.zeros(hidden_size)

for t, X in enumerate(X_seq, 1):
    print(f"\n--- t={t} ---")
    show("X_t",      X)
    show("H_{t-1}",  H)
    show("C_{t-1}",  C)

    F       = sigmoid(X @ W_xf.T + H @ W_hf.T + b_f)
    I       = sigmoid(X @ W_xi.T + H @ W_hi.T + b_i)
    C_tilde = tanh(   X @ W_xc.T + H @ W_hc.T + b_c)
    C_new   = F * C + I * C_tilde
    O       = sigmoid(X @ W_xo.T + H @ W_ho.T + b_o)
    H_new   = O * tanh(C_new)

    show("F_t  (forget gate)",  F)
    show("I_t  (input  gate)",  I)
    show("C_tilde (candidate)", C_tilde)
    show("C_t  (cell state)",   C_new)
    show("O_t  (output gate)",  O)
    show("H_t  (hidden state)", H_new)

    H, C = H_new, C_new

print("\n" + "="*50)
print("  FINAL STATES")
print("="*50)
show("H_t (short memory)", H)
show("C_t (long  memory)", C)
