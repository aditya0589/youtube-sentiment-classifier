import sys
# Filter out global Spark paths to prevent compatibility errors with newer Python versions on startup
sys.path = [p for p in sys.path if "spark" not in p.lower()]

import os
import re
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.model_predictor import ModelPredictor
from src.components.database import (
    init_db, save_video_to_cache, get_video_from_cache,
    get_all_cached_videos, log_metric, get_aggregated_metrics, clear_all_data
)
from main import run_pipeline

app = FastAPI(title="YouTube Sentiment Analyzer API")

@app.on_event("startup")
async def startup_event():
    init_db()

# Enable CORS for cross-origin frontend queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Latency and API Metrics Middleware
@app.middleware("http")
async def log_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    path = request.url.path
    # Only track API endpoints to keep metric logs clean
    if path.startswith(("/predict", "/analyze-video", "/retrain", "/api")):
        try:
            log_metric(path, process_time, response.status_code)
        except Exception:
            pass
            
    return response

# Pydantic request body validations
class PredictionRequest(BaseModel):
    text: str

class VideoAnalysisRequest(BaseModel):
    url: str

class RetrainRequest(BaseModel):
    channel_ids: List[str] = ["UC-lHJZR3Gqxm24_Vd_AJ5Yw", "UC16niRr50-MSBwiO3YDb3RA"]

class ClearCacheRequest(BaseModel):
    password: str

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
        
        # 1. Check cache first
        cached = get_video_from_cache(video_id)
        if cached:
            return JSONResponse(content={
                "video_id": video_id,
                "video_title": cached["video_title"],
                "summary": cached["summary"],
                "comments": cached["comments"],
                "cached": True
            })
            
        # 2. Fetch Video Details (to get Title)
        ingestor = DataIngestion()
        video_title = f"YouTube Video ({video_id})"
        try:
            video_details = ingestor.get_video_details([video_id])
            if video_details:
                video_title = video_details[0]['snippet']['title']
        except Exception as title_err:
            # Non-blocking fallback to ID
            pass
            
        # 3. Fetch Comments
        # Limit to 1 page (100 comments) for fast real-time API response
        comments = ingestor.get_comments(video_id=video_id, max_results=100, page_limit=1)
        if not comments:
            return JSONResponse(status_code=400, content={"error": "No comments found on this video."})
            
        # 4. Clean Comments
        cleaning = DataCleaning()
        df_comments = ingestor.convert_comments_to_dataframe(comments)
        df_clean = cleaning.clean_data(df_comments)
        df_clean = cleaning.remove_non_english(df_clean)
        df_clean = cleaning.remove_symbols(df_clean)
        
        text_column = 'snippet.topLevelComment.snippet.textOriginal'
        if text_column not in df_clean.columns or len(df_clean) == 0:
            return JSONResponse(status_code=400, content={"error": "No clean English comments available."})
            
        texts = df_clean[text_column].tolist()
        
        # 5. Classify Sentiments using our Custom ML model
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
        
        # 6. Save to cache
        save_video_to_cache(video_id, video_title, summary, results)
        
        return JSONResponse(content={
            "video_id": video_id,
            "video_title": video_title,
            "summary": summary,
            "comments": results[:50],  # Return up to 50 comments for displaying in feed
            "cached": False
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    """
    Trigger the model retraining pipeline on fresh comments from a list of YouTube channels.
    """
    try:
        # Run the ingestion + transformation + training pipeline in-memory
        run_pipeline(channel_ids=request.channel_ids)
        
        # Reload the newly saved model/vectorizer in-memory
        global predictor
        predictor = ModelPredictor()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Model retrained and reloaded in-memory successfully!",
            "channels_processed": request.channel_ids
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Retraining failed: {str(e)}"})

@app.get("/api/history")
async def get_history():
    try:
        history = get_all_cached_videos()
        return JSONResponse(content={"history": history})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/history/{video_id}")
async def get_history_detail(video_id: str):
    try:
        cached = get_video_from_cache(video_id)
        if not cached:
            return JSONResponse(status_code=404, content={"error": "Analysis not found in cache."})
        return JSONResponse(content=cached)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/metrics")
async def get_metrics():
    try:
        metrics = get_aggregated_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/model-info")
async def get_model_info():
    try:
        model_name = type(predictor.model).__name__
        params = predictor.model.get_params()
        vocab_size = len(predictor.vectorizer.vocabulary_) if hasattr(predictor.vectorizer, 'vocabulary_') else 0
        return JSONResponse(content={
            "model_name": model_name,
            "vocab_size": vocab_size,
            "hyperparameters": {
                "C": params.get("C"),
                "class_weight": str(params.get("class_weight")),
                "max_iter": params.get("max_iter"),
                "solver": params.get("solver")
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to retrieve model info: {str(e)}"})

@app.post("/api/clear-cache")
async def clear_cache(request: ClearCacheRequest):
    try:
        expected_password = os.getenv("ADMIN_PASSWORD", "admin")
        if request.password != expected_password:
            return JSONResponse(status_code=401, content={"error": "Invalid admin password."})
        clear_all_data()
        return JSONResponse(content={"status": "success", "message": "Cache and metrics cleared successfully!"})
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
