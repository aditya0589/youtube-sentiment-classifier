import sys
# Filter out global Spark paths to prevent compatibility errors with newer Python versions on startup
sys.path = [p for p in sys.path if "spark" not in p.lower()]

import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.model_predictor import ModelPredictor

app = FastAPI(title="YouTube Sentiment Analyzer API")

# Enable CORS for cross-origin frontend queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request body validations
class PredictionRequest(BaseModel):
    text: str

class VideoAnalysisRequest(BaseModel):
    url: str

def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL (both desktop and mobile formats).
    """
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return url.strip()

# Initialize predictor (loads model.pkl and vectorizer.pkl)
predictor = ModelPredictor()

@app.post("/predict")
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text comment.
    """
    try:
        sentiment = predictor.predict(request.text)
        return JSONResponse(content={"text": request.text, "sentiment": sentiment})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Ingest, clean, and run predictions on comments from a YouTube video URL in real-time.
    """
    try:
        video_id = extract_video_id(request.url)
        
        # 1. Fetch Comments
        ingestor = DataIngestion()
        # Limit to 1 page (100 comments) for fast real-time API response
        comments = ingestor.get_comments(video_id=video_id, max_results=100, page_limit=1)
        if not comments:
            return JSONResponse(status_code=400, content={"error": "No comments found on this video."})
            
        # 2. Clean Comments
        cleaning = DataCleaning()
        df_comments = ingestor.convert_comments_to_dataframe(comments)
        df_clean = cleaning.clean_data(df_comments)
        df_clean = cleaning.remove_non_english(df_clean)
        df_clean = cleaning.remove_symbols(df_clean)
        
        text_column = 'snippet.topLevelComment.snippet.textOriginal'
        if text_column not in df_clean.columns or len(df_clean) == 0:
            return JSONResponse(status_code=400, content={"error": "No clean English comments available."})
            
        texts = df_clean[text_column].tolist()
        
        # 3. Classify Sentiments using our Custom ML model
        results = []
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        
        for text in texts:
            sentiment = predictor.predict(text)
            results.append({"text": text, "sentiment": sentiment})
            if sentiment == 'positive':
                positive_count += 1
            elif sentiment == 'neutral':
                neutral_count += 1
            elif sentiment == 'negative':
                negative_count += 1
                
        total = len(texts)
        summary = {
            "total": total,
            "positive": {
                "count": positive_count,
                "percentage": round((positive_count / total) * 100, 2) if total > 0 else 0
            },
            "neutral": {
                "count": neutral_count,
                "percentage": round((neutral_count / total) * 100, 2) if total > 0 else 0
            },
            "negative": {
                "count": negative_count,
                "percentage": round((negative_count / total) * 100, 2) if total > 0 else 0
            }
        }
        
        return JSONResponse(content={
            "video_id": video_id,
            "summary": summary,
            "comments": results[:50]  # Return up to 50 comments for displaying in feed
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """
    Serve the frontend dashboard HTML.
    """
    template_path = os.path.join("templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>YouTube Sentiment Analyzer API is Live! index.html missing.</h3>"
