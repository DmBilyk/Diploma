"""
lstm_model.py
=============
LSTM neural network for multi-asset return prediction.
Built with TensorFlow / Keras.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class PortfolioLSTMModel:
    """Sequence-to-vector LSTM that predicts next-step log-returns
    for every asset simultaneously.

    Input shape : (batch, seq_length, num_assets)
    Output shape: (batch, num_assets)
    """

    def __init__(
        self,
        seq_length: int,
        num_assets: int,
        lstm_units: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        self.seq_length = seq_length
        self.num_assets = num_assets
        self.model = self._build(lstm_units, dropout, learning_rate)

    # ------------------------------------------------------------------
    #  Architecture
    # ------------------------------------------------------------------
    def _build(
        self,
        lstm_units: Tuple[int, ...],
        dropout: float,
        learning_rate: float,
    ) -> keras.Model:
        model = keras.Sequential(name="portfolio_lstm")
        for i, units in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            model.add(
                keras.layers.LSTM(
                    units,
                    return_sequences=return_seq,
                    input_shape=(self.seq_length, self.num_assets) if i == 0 else None,
                )
            )
            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(self.num_assets))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        logger.info("LSTM model built: %s", model.summary(print_fn=lambda x: x))
        return model

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
    ) -> keras.callbacks.History:
        """Train the model. Returns Keras History object."""
        validation_data = (X_val, y_val) if X_val is not None else None
        return self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=0,
        )

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-step returns. Returns shape (samples, num_assets)."""
        return self.model.predict(X, verbose=0)

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "PortfolioLSTMModel":
        """Load a saved Keras model and wrap it."""
        loaded = keras.models.load_model(path)
        _, seq_length, num_assets = loaded.input_shape
        instance = cls.__new__(cls)
        instance.seq_length = seq_length
        instance.num_assets = num_assets
        instance.model = loaded
        return instance
