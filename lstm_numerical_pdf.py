import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    def __init__(self, Wf, Whf, bf, Wi, Whi, bi, Wc, Whc, bc, Wo, Who, bo):
        self.Wf, self.Whf, self.bf = Wf, Whf, bf
        self.Wi, self.Whi, self.bi = Wi, Whi, bi
        self.Wc, self.Whc, self.bc = Wc, Whc, bc
        self.Wo, self.Who, self.bo = Wo, Who, bo

    def forward(self, x, h_prev, C_prev):
        f       = sigmoid(self.Wf * x + self.Whf * h_prev + self.bf)
        i       = sigmoid(self.Wi * x + self.Whi * h_prev + self.bi)
        C_tilde = np.tanh(self.Wc  * x + self.Whc * h_prev + self.bc)
        C       = f * C_prev + i * C_tilde
        o       = sigmoid(self.Wo * x + self.Who * h_prev + self.bo)
        h       = o * np.tanh(C)
        return h, C


class LSTMModel:
    def __init__(self, cell, Wy, by):
        self.cell = cell
        self.Wy   = Wy
        self.by   = by

    def predict(self, sequence, h0=0.0, C0=0.0):
        h, C = h0, C0
        for x in sequence:
            h, C = self.cell.forward(x, h, C)
        return self.Wy * h + self.by


# ── Build model with PDF weights ─────────────────────────────
cell = LSTMCell(
    Wf=0.5, Whf=0.1, bf=0,
    Wi=0.6, Whi=0.2, bi=0,
    Wc=0.7, Whc=0.3, bc=0,
    Wo=0.8, Who=0.4, bo=0,
)
model = LSTMModel(cell, Wy=4, by=0)

# ── Run ──────────────────────────────────────────────────────
X     = [1, 2, 3, 4]
y_hat = model.predict(X)

print(f"Input sequence : {X}")
print(f"Predicted next : {round(y_hat, 4)}")
print(f"Target         : 4")
print(f"Error          : {round(abs(y_hat - 4), 4)}")
