# CNN-LSTM Sign Language Recognition

This project adapts the same CNN-LSTM architecture style used in HAR to a new task:

- Application: Sign Language Recognition
- Input: temporal hand landmark sequences `(64, 63)`
- Model flow:
  - Conv1D(64) -> BatchNorm -> ReLU -> MaxPool
  - Conv1D(128) -> BatchNorm -> ReLU -> MaxPool
  - LSTM(128) -> Dropout
  - LSTM(64) -> Dropout
  - Dense(128) -> Dropout
  - Softmax output

## Folder Structure

- `main.py`: training + evaluation pipeline
- `requirements.txt`: dependencies
- `data/`: optional custom dataset input (`sign_sequences.npz`)
- `artifacts/`: saved model and metrics

## Use Your Own Data (Optional)

Place this file at:

- `data/sign_sequences.npz`

Expected arrays inside NPZ:

- `X`: shape `(samples, 64, 63)`
- `y`: shape `(samples,)`, integer labels

If NPZ is not found, the script automatically generates synthetic training data.

## Run

```bash
cd D:\Desktop\CNN-LSTM-Sign-Language-Recognition
pip install -r requirements.txt
python main.py
```

Outputs will be saved in `artifacts/`:

- best model checkpoint
- final model
- training history
- metrics and confusion matrix JSON
