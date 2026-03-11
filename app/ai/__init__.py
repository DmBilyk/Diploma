"""
app.ai — LSTM-based AI module for portfolio return prediction.

Public API
----------
PortfolioLSTMModel       – TensorFlow/Keras LSTM model
TrainingWorker           – QThread for non-blocking training
predict_expected_returns – generate μ vector from a trained model
run_lstm_optimization    – end-to-end LSTM → evolutionary optimizer pipeline
"""

from ai.lstm_model import PortfolioLSTMModel
from ai.training_worker import TrainingWorker
from ai.predictor import predict_expected_returns, run_lstm_optimization
