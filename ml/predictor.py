from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class StockPredictor:
    """
    A class to predict stock movements based on historical price data and sentiment analysis.
    Uses a Random Forest model by default.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the stock predictor.

        Args:
            model_path (str): Path to a saved model file. If None, a new model is initialized.
        """
        self.scaler = StandardScaler()
        self._is_trained = False

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(model_path.replace(".pkl", "_scaler.pkl"))
                self._is_trained = True
                print("Model and scaler loaded successfully.")
            except Exception as e:
                print(f"Failed to load saved model. Reinitializing. Error: {e}")
                self._init_new_model()
        else:
            self._init_new_model()

    def _init_new_model(self):
        """Initialize a new Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self._is_trained = False

    def prepare_features(self, price_data: pd.DataFrame, news_sentiment: float = 0.0) -> pd.DataFrame:
        """
        Prepare feature set from price data and sentiment.

        Args:
            price_data (pd.DataFrame): Historical stock data with 'Close' and 'Volume'.
            news_sentiment (float): Aggregated news sentiment score (-1 to 1).

        Returns:
            pd.DataFrame: Latest row of processed features ready for prediction.
        """
        df = price_data.copy()

        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Volatility'] = df['Price_Change'].rolling(5).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['RSI'] = self._calculate_rsi(df['Close'], window=14)
        df['News_Sentiment'] = news_sentiment

        df = df.dropna()

        features = [
            'MA5', 'MA10', 'MA20', 'Price_Change', 'Price_Change_5d',
            'Volatility', 'Volume_Change', 'Volume_MA5', 'RSI', 'News_Sentiment'
        ]

        if df.empty:
            return pd.DataFrame([[0] * len(features)], columns=features)

        return df[features].iloc[-1:].copy()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).

        Args:
            prices (pd.Series): Series of closing prices.
            window (int): Window size for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model using historical features and labels.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Target labels (1: up, 0: down).
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._is_trained = True
        print("Model trained successfully.")

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict the stock movement.

        Args:
            features (pd.DataFrame): Processed features from prepare_features().

        Returns:
            Tuple[str, float]: (prediction, confidence) — 'up' or 'down' and its probability.
        """
        if not self._is_trained:
            recent_change = features.get('Price_Change', pd.Series([0])).values[0]
            sentiment = features.get('News_Sentiment', pd.Series([0])).values[0]
            combined_signal = recent_change + (sentiment * 0.5)
            prediction = "up" if combined_signal > 0 else "down"
            confidence = min(abs(combined_signal) * 2, 0.75)
            return prediction, float(confidence)

        X_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(X_scaled)[0]
        predicted_class_idx = np.argmax(probabilities)
        prediction = "up" if predicted_class_idx == 1 else "down"
        confidence = float(probabilities[predicted_class_idx])
        return prediction, confidence

    def save_model(self, model_path: str):
        """
        Save trained model and scaler to disk.

        Args:
            model_path (str): Path where model will be saved.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving.")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_path.replace(".pkl", "_scaler.pkl"))
        print(f"Model saved to {model_path}")

