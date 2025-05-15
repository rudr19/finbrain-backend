import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, Tuple, List

class StockPredictor:
    """
    A class to predict stock movements based on historical price data and sentiment analysis.
    Uses a Random Forest model by default, but can be extended to use more sophisticated models.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the stock predictor.
        
        Args:
            model_path: Path to a saved model file. If None, a new model will be trained.
        """
        self.scaler = StandardScaler()
        
        # Try to load a pretrained model if it exists
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
                self._is_trained = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self._init_new_model()
        else:
            self._init_new_model()
    
    def _init_new_model(self):
        """Initialize a new model when no pretrained model is available."""
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self._is_trained = False
    
    def prepare_features(self, price_data: pd.DataFrame, news_sentiment: float = 0) -> pd.DataFrame:
        """
        Prepare features for prediction from raw price data and sentiment.
        
        Args:
            price_data: DataFrame with OHLCV data
            news_sentiment: Average sentiment score from news (-1 to 1)
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original data
        df = price_data.copy()
        
        # Calculate basic technical indicators
        # 1. Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 2. Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # 3. Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=5).std()
        
        # 4. Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # 5. Price momentum
        df['RSI'] = self._calculate_rsi(df['Close'], window=14)
        
        # Add sentiment data
        df['News_Sentiment'] = news_sentiment
        
        # Drop rows with NaN values (resulting from rolling calculations)
        df = df.dropna()
        
        # Select features for the model
        features = [
            'MA5', 'MA10', 'MA20', 'Price_Change', 'Price_Change_5d',
            'Volatility', 'Volume_Change', 'Volume_MA5', 'RSI', 'News_Sentiment'
        ]
        
        # If the DataFrame is empty after processing, return a minimal feature set
        if df.empty:
            # Create a single row of zeros for all features
            return pd.DataFrame([[0] * len(features)], columns=features)
        
        return df[features].iloc[-1:].copy()  # Return the most recent data point
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model on historical data.
        
        Args:
            X_train: Feature DataFrame
            y_train: Target Series (1 for up, 0 for down)
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_scaled, y_train)
        self._is_trained = True
    
    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """
        Make a prediction for stock movement.
        
        Args:
            features: DataFrame with features calculated by prepare_features
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: "up" or "down"
            confidence: float value between 0 and 1
        """
        # If the model isn't trained yet, make a pseudo-prediction based on simple logic
        if not self._is_trained:
            # Default prediction logic: if recent price trend and sentiment are positive, predict up
            recent_change = features['Price_Change'].values[0] if 'Price_Change' in features else 0
            sentiment = features['News_Sentiment'].values[0] if 'News_Sentiment' in features else 0
            
            # Simple heuristic for prediction when model isn't trained
            combined_signal = recent_change + (sentiment * 0.5)
            prediction = "up" if combined_signal > 0 else "down"
            confidence = min(abs(combined_signal) * 2, 0.75)  # Cap confidence at 75% for heuristic predictions
            
            return prediction, float(confidence)
        
        # Scale the features
        X_scaled = self.scaler.transform(features)
        
        # Get class probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get the index of the class with the highest probability
        predicted_class_idx = np.argmax(probabilities)
        
        # Map to "up" or "down"
        prediction = "up" if predicted_class_idx == 1 else "down"
        
        # Confidence is the probability of the predicted class
        confidence = float(probabilities[predicted_class_idx])
        
        return prediction, confidence
    
    def save_model(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where the model will be saved
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_path.replace('.pkl', '_scaler.pkl'))
