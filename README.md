# RNN & LSTM — From Scratch (Python)
**Based on Lecture 3 by A. Prof. Noha El-Attar**

---

## How to Run

```bash
pip install numpy
python lstm_from_scratch.py
```

---

## What the Script Does

### Part 1 — Vanilla RNN
Replicates the **"dogs"** example from the lecture slides.

- Input sequence: `d → o → g`
- Target prediction: `s`
- Vocabulary size `k = 4`, Hidden nodes `d = 3`
- Uses the exact weight matrices from the slides

**Equations:**
```
a_t = W_H · h_{t-1}  +  W_X · X_t
h_t = tanh(a_t)
y_t = softmax(W_Y · h_t)
E   = (ŷ - y)²
```

---

### Part 2 — LSTM Forward Pass
Full implementation of an LSTM cell with **4 gates** across 3 time steps.

**Gate Equations:**

| Gate | Equation | Role |
|------|----------|------|
| Forget gate | `F_t = σ(X_t·W_xf + H_{t-1}·W_hf + b_f)` | What to erase from long memory |
| Input gate | `I_t = σ(X_t·W_xi + H_{t-1}·W_hi + b_i)` | What new info to write |
| Candidate | `C̃_t = tanh(X_t·W_xc + H_{t-1}·W_hc + b_c)` | New candidate values |
| Cell state | `C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t` | Updated long memory |
| Output gate | `O_t = σ(X_t·W_xo + H_{t-1}·W_ho + b_o)` | What to output |
| Hidden state | `H_t = O_t ⊙ tanh(C_t)` | Short memory / output |

---

## Key Concepts

### RNN vs LSTM

| | RNN | LSTM |
|---|---|---|
| Architecture | Single tanh layer | 4 interactive layers |
| Memory | Short-term only | Long + short term |
| Gradient problem | Vanishing gradients | Solved via cell state |

### Why LSTM Solves Vanishing Gradients
The **cell state** `C_t` uses **additive updates** instead of repeated multiplications, allowing gradients to flow back through long sequences without shrinking to zero.

### LSTM Memory Types
- **Long memory** → Cell state `C_t` (runs across all time steps like a conveyor belt)
- **Short memory** → Hidden state `H_t` (output at each time step)


Gate	Wx	Wh	b
Forget	0.5	0.1	0
Input	0.6	0.2	0
Candidate	0.7	0.3	0
Output	0.8	0.4	0


t	f	i	C̃	C	o	h
1	0.622	0.646	0.604	0.390	0.690	0.257
2	0.736	0.778	0.901	0.988	0.846	0.640
3	0.827	0.873	0.980	1.672	0.934	0.871
4	0.889	0.929	0.996	2.390	0.965	0.949


