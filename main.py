"""
============================================================
  CNN-LSTM Sign Language Recognition (SLR)
  Project: Deep Learning Assignment
  Model: CNN-LSTM Hybrid Architecture
  Dataset: Synthetic landmark sequences (or custom NPZ)
============================================================

DESCRIPTION:
  This project reuses the same HAR-style CNN-LSTM idea for
  sign language recognition using temporal hand landmarks.

  CNN extracts local motion patterns from short windows.
  LSTM captures long-range temporal dependencies between windows.

INPUT FORMAT:
  - Each sample is a sequence of hand landmarks:
    (timesteps, features) = (64, 63)
  - 63 = 21 landmarks x (x, y, z)

CLASSES (default synthetic setup):
  0 - Hello
  1 - Thanks
  2 - Yes
  3 - No
  4 - I Love You
  5 - Please
  6 - Sorry
  7 - Help
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42

CONFIG = {
    "timesteps": 64,
    "features": 63,
    "n_classes": 8,
    "n_samples": 4800,
    "test_size": 0.20,
    "val_size": 0.20,
    "cnn_filters_1": 64,
    "cnn_filters_2": 128,
    "kernel_size": 3,
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 128,
    "dropout_cnn": 0.2,
    "dropout_lstm": 0.3,
    "dropout_dense": 0.4,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 25,
    "patience": 5,
}

SIGN_LABELS = {
    0: "Hello",
    1: "Thanks",
    2: "Yes",
    3: "No",
    4: "I Love You",
    5: "Please",
    6: "Sorry",
    7: "Help",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def add_axis(seq: np.ndarray, landmark_idx: int, axis: int, signal: np.ndarray) -> None:
    seq[:, landmark_idx * 3 + axis] += signal


def add_to_tips(seq: np.ndarray, axis: int, signal: np.ndarray) -> None:
    for lm in [4, 8, 12, 16, 20]:
        add_axis(seq, lm, axis, signal)


def apply_sign_pattern(seq: np.ndarray, label: int, t: np.ndarray) -> None:
    # Wrist indices: landmark 0 -> x:0, y:1, z:2
    if label == 0:  # Hello: side wave
        add_axis(seq, 0, 0, 0.30 * np.sin(2 * np.pi * 2.2 * t))
        add_to_tips(seq, 1, 0.06 * np.sin(2 * np.pi * 4.0 * t))

    elif label == 1:  # Thanks: outward motion
        add_axis(seq, 0, 2, np.linspace(0.25, -0.20, len(t)))
        add_to_tips(seq, 0, np.linspace(0.0, 0.15, len(t)))

    elif label == 2:  # Yes: short vertical nods
        add_axis(seq, 0, 1, 0.22 * np.sin(2 * np.pi * 3.0 * t))
        add_to_tips(seq, 1, 0.10 * np.sin(2 * np.pi * 3.0 * t + 0.6))

    elif label == 3:  # No: side-to-side shake
        add_axis(seq, 0, 0, 0.26 * np.sin(2 * np.pi * 2.8 * t + np.pi / 3))
        add_to_tips(seq, 0, 0.12 * np.sin(2 * np.pi * 2.8 * t + np.pi / 2))

    elif label == 4:  # I Love You: distinct fingers emphasized
        add_axis(seq, 4, 1, 0.16 * np.ones_like(t))
        add_axis(seq, 8, 1, 0.20 * np.ones_like(t))
        add_axis(seq, 20, 1, 0.20 * np.ones_like(t))
        add_axis(seq, 0, 0, 0.08 * np.sin(2 * np.pi * 1.2 * t))

    elif label == 5:  # Please: circular motion near chest
        add_axis(seq, 0, 0, 0.18 * np.cos(2 * np.pi * 1.6 * t))
        add_axis(seq, 0, 1, 0.18 * np.sin(2 * np.pi * 1.6 * t))

    elif label == 6:  # Sorry: small circular fist motion
        add_axis(seq, 0, 0, 0.10 * np.cos(2 * np.pi * 2.0 * t))
        add_axis(seq, 0, 1, 0.10 * np.sin(2 * np.pi * 2.0 * t))
        add_to_tips(seq, 2, -0.06 * np.ones_like(t))

    else:  # label == 7, Help: upward then hold
        rise = np.clip(3.5 * t, 0.0, 1.0)
        add_axis(seq, 0, 1, 0.30 * rise)
        add_to_tips(seq, 1, 0.14 * rise)


def generate_synthetic_slr_data(
    n_samples: int,
    n_classes: int,
    timesteps: int,
    features: int,
    seed: int = 42,
):
    np.random.seed(seed)
    X = np.zeros((n_samples, timesteps, features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)

    t = np.linspace(0.0, 1.0, timesteps).astype(np.float32)

    for i in range(n_samples):
        label = i % n_classes
        y[i] = label

        seq = np.random.normal(0.0, 0.015, size=(timesteps, features)).astype(np.float32)

        # Base hand geometry trend across 21 landmarks.
        for lm in range(21):
            seq[:, lm * 3 + 0] += (lm % 5) * 0.02
            seq[:, lm * 3 + 1] += (lm // 5) * 0.02
            seq[:, lm * 3 + 2] += (lm % 3) * 0.01

        local_t = np.clip(t + np.random.uniform(-0.07, 0.07), 0.0, 1.0)
        apply_sign_pattern(seq, label, local_t)

        amplitude = np.random.uniform(0.85, 1.15)
        X[i] = seq * amplitude

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def normalize_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes=n_classes).astype(np.float32)


def build_cnn_lstm_model(config: dict) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(config["timesteps"], config["features"]))

    x = tf.keras.layers.Conv1D(config["cnn_filters_1"], config["kernel_size"], padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(config["dropout_cnn"])(x)

    x = tf.keras.layers.Conv1D(config["cnn_filters_2"], config["kernel_size"], padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(config["dropout_cnn"])(x)

    x = tf.keras.layers.LSTM(config["lstm_units_1"], return_sequences=True)(x)
    x = tf.keras.layers.Dropout(config["dropout_lstm"])(x)

    x = tf.keras.layers.LSTM(config["lstm_units_2"], return_sequences=False)(x)
    x = tf.keras.layers.Dropout(config["dropout_lstm"])(x)

    x = tf.keras.layers.Dense(config["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(config["dropout_dense"])(x)
    outputs = tf.keras.layers.Dense(config["n_classes"], activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_sign_language")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_or_generate_data(project_root: Path, config: dict):
    data_path = project_root / "data" / "sign_sequences.npz"

    if data_path.exists():
        data = np.load(data_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)

        if X.ndim != 3:
            raise ValueError("Expected X with shape (samples, timesteps, features).")
        if X.shape[1] != config["timesteps"] or X.shape[2] != config["features"]:
            raise ValueError(
                f"Expected X shape (*, {config['timesteps']}, {config['features']}). Got {X.shape}."
            )

        source = "custom_npz"
    else:
        X, y = generate_synthetic_slr_data(
            n_samples=config["n_samples"],
            n_classes=config["n_classes"],
            timesteps=config["timesteps"],
            features=config["features"],
            seed=SEED,
        )
        source = "synthetic"

    return X, y, source


def main():
    project_root = Path(__file__).resolve().parent
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)

    print("=" * 68)
    print("  CNN-LSTM Sign Language Recognition")
    print("=" * 68)

    X, y, source = load_or_generate_data(project_root, CONFIG)
    n_classes = int(np.max(y)) + 1

    run_config = dict(CONFIG)
    run_config["n_classes"] = n_classes

    class_names = [SIGN_LABELS.get(i, f"Class_{i}") for i in range(n_classes)]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=run_config["test_size"],
        random_state=SEED,
        stratify=y,
    )

    val_ratio = run_config["val_size"] / (1.0 - run_config["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=SEED,
        stratify=y_train_val,
    )

    X_train, X_val, X_test, mean, std = normalize_splits(X_train, X_val, X_test)

    y_train_oh = to_one_hot(y_train, run_config["n_classes"])
    y_val_oh = to_one_hot(y_val, run_config["n_classes"])
    y_test_oh = to_one_hot(y_test, run_config["n_classes"])

    print(f"Data source      : {source}")
    print(f"Train shape      : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Test shape       : {X_test.shape}")
    print(f"Classes          : {class_names}")

    model = build_cnn_lstm_model(run_config)
    model.summary()

    best_model_path = artifacts_dir / "best_cnn_lstm_sign_language.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=run_config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=run_config["epochs"],
        batch_size=run_config["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    eval_metrics = model.evaluate(X_test, y_test_oh, verbose=0, return_dict=True)
    if not isinstance(eval_metrics, dict):
        raise TypeError("Expected evaluate(..., return_dict=True) to return a dict.")

    test_loss = float(eval_metrics.get("loss", 0.0))
    test_acc = float(eval_metrics.get("accuracy", 0.0))
    test_auc = float(eval_metrics.get("auc", 0.0))
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    model.save(artifacts_dir / "final_cnn_lstm_sign_language.keras")

    history_path = artifacts_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    results = {
        "project": "CNN-LSTM Sign Language Recognition",
        "data_source": source,
        "config": run_config,
        "class_names": class_names,
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_auc": test_auc,
        },
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    results_path = artifacts_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Test AUC  : {test_auc:.4f}")
    print("Classification Report:")
    print(report_text)
    print(f"Saved best model : {best_model_path}")
    print(f"Saved history    : {history_path}")
    print(f"Saved results    : {results_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
