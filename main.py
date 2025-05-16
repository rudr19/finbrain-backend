from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from pydantic import BaseModel
import os
from typing import List, Dict, Any, Optional
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ml.predictor import StockPredictor

# Initialize NLTK components
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "2cdc5797e7fd4753b24e7b80d62068de")

# Initialize the FastAPI app
app = FastAPI(title="FinBrain API", description="Stock Sentiment and Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize stock predictor model
predictor = StockPredictor()

class PredictionResponse(BaseModel):
    ticker: str
    prediction: str  # "up" or "down"
    confidence: float
    price_signals: Dict[str, Any]
    sentiment_signals: Dict[str, Any]
    news_signals: Dict[str, Any]

class SentimentResponse(BaseModel):
    ticker: str
    overall_sentiment: float
    reddit_sentiment: Optional[float]
    twitter_sentiment: Optional[float]
    news_sentiment: float
    sentiment_timeline: List[Dict[str, Any]]
    news_highlights: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"message": "Welcome to FinBrain API!"}

# Original routes
@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, days: int = 30):
    """Get historical stock data for the specified ticker."""
    return _get_stock_data(ticker, days)

@app.get("/news/{ticker}")
def get_news(ticker: str, days: int = 7):
    """Get recent news articles related to the stock."""
    return _get_news(ticker, days)

@app.post("/predict")
def predict_stock_movement(ticker: str = Query(..., description="Stock ticker symbol")):
    """Predict stock movement using multi-modal data."""
    return _predict_stock_movement(ticker)

@app.get("/sentiment")
def get_sentiment(ticker: str = Query(..., description="Stock ticker symbol")):
    """Get sentiment analysis for a stock."""
    return _get_sentiment(ticker)

# New routes with /api prefix matching frontend requests
@app.get("/api/stocks/{ticker}")
def api_get_stock_data(ticker: str, days: int = 30):
    """API route for getting historical stock data."""
    return _get_stock_data(ticker, days)

@app.get("/api/news/{ticker}")
def api_get_news(ticker: str, days: int = 7):
    """API route for getting news data."""
    return _get_news(ticker, days)

@app.get("/api/prediction/{ticker}")
def api_predict_stock_movement(ticker: str):
    """API route for predicting stock movement."""
    return _predict_stock_movement(ticker)

@app.get("/api/sentiment/{ticker}")
def api_get_sentiment(ticker: str):
    """API route for getting sentiment analysis."""
    return _get_sentiment(ticker)

# Helper functions to avoid code duplication
def _get_stock_data(ticker: str, days: int = 30):
    """Helper function for stock data."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Format the response
        data = []
        for index, row in hist.iterrows():
            data.append({
                "date": index.strftime('%Y-%m-%d'),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": row['Volume']
            })
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_news(ticker: str, days: int = 7):
    """Helper function for news data."""
    try:
        # Get company name for better news search
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query NewsAPI
        url = f"https://newsapi.org/v2/everything?q={company_name}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        
        response = requests.get(url)
        news_data = response.json()
        
        articles = []
        for article in news_data.get('articles', [])[:10]:  # Limit to top 10 articles
            # Process article text and calculate sentiment
            text = article.get('title', '') + '. ' + article.get('description', '')
            sentiment = sia.polarity_scores(text)
            
            articles.append({
                "title": article.get('title', ''),
                "description": article.get('description', ''),
                "source": article.get('source', {}).get('name', ''),
                "url": article.get('url', ''),
                "publishedAt": article.get('publishedAt', ''),
                "sentiment": {
                    "positive": sentiment['pos'],
                    "negative": sentiment['neg'],
                    "neutral": sentiment['neu'],
                    "compound": sentiment['compound']
                }
            })
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "articles": articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _predict_stock_movement(ticker: str):
    """Helper function for stock prediction."""
    try:
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Get news and sentiment data
        news_response = _get_news(ticker)
        news_sentiment = [article["sentiment"]["compound"] for article in news_response["articles"]]
        avg_news_sentiment = sum(news_sentiment) / len(news_sentiment) if news_sentiment else 0
        
        # Prepare features for prediction
        features = predictor.prepare_features(hist, avg_news_sentiment)
        
        # Make prediction
        prediction, confidence = predictor.predict(features)
        
        # Create response with signals
        price_signals = {
            "trend": "up" if hist["Close"].pct_change().mean() > 0 else "down",
            "volatility": hist["Close"].pct_change().std(),
            "last_change": hist["Close"].pct_change().iloc[-1] if len(hist) > 1 else 0
        }
        
        sentiment_signals = {
            "twitter": 0,  # Placeholder for Twitter sentiment
            "reddit": 0,   # Placeholder for Reddit sentiment
        }
        
        news_signals = {
            "avg_sentiment": avg_news_sentiment,
            "article_count": len(news_response["articles"])
        }
        
        return PredictionResponse(
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            price_signals=price_signals,
            sentiment_signals=sentiment_signals,
            news_signals=news_signals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_sentiment(ticker: str):
    """Helper function for sentiment analysis."""
    try:
        # Get news sentiment
        news_response = _get_news(ticker)
        news_sentiment = [article["sentiment"]["compound"] for article in news_response["articles"]]
        avg_news_sentiment = sum(news_sentiment) / len(news_sentiment) if news_sentiment else 0
        
        # Create sentiment timeline (simplified)
        timeline = []
        for i, article in enumerate(news_response["articles"]):
            pub_date = article.get("publishedAt", "")
            if pub_date:
                timeline.append({
                    "date": pub_date[:10],  # Extract just the date part
                    "sentiment": article["sentiment"]["compound"]
                })
        
        # Sort by date
        timeline.sort(key=lambda x: x["date"])
        
        # Select top positive and negative news as highlights
        sorted_articles = sorted(news_response["articles"], 
                               key=lambda x: x["sentiment"]["compound"])
        
        highlights = []
        # Add most negative articles
        if len(sorted_articles) > 0:
            highlights.append(sorted_articles[0])
        # Add most positive articles
        if len(sorted_articles) > 1:
            highlights.append(sorted_articles[-1])
        
        return SentimentResponse(
            ticker=ticker,
            overall_sentiment=avg_news_sentiment,
            reddit_sentiment=None,  # Placeholder for Reddit sentiment
            twitter_sentiment=None, # Placeholder for Twitter sentiment
            news_sentiment=avg_news_sentiment,
            sentiment_timeline=timeline,
            news_highlights=[{
                "title": h["title"],
                "description": h["description"],
                "source": h["source"],
                "url": h["url"],
                "sentiment": h["sentiment"]["compound"]
            } for h in highlights]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
