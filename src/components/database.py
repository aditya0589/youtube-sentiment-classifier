import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "data/sentiment_cache.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Table for cached video analyses
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_cache (
            video_id TEXT PRIMARY KEY,
            video_title TEXT,
            summary TEXT,
            comments TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. Table for system latency / metrics logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT,
            latency REAL,
            status_code INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. Table for YouTube API quota logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quota_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT,
            units_consumed INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_video_to_cache(video_id, video_title, summary_dict, comments_list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO video_cache (video_id, video_title, summary, comments, analyzed_at) VALUES (?, ?, ?, ?, ?)",
        (video_id, video_title, json.dumps(summary_dict), json.dumps(comments_list), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_video_from_cache(video_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT video_title, summary, comments FROM video_cache WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "video_id": video_id,
            "video_title": row[0],
            "summary": json.loads(row[1]),
            "comments": json.loads(row[2])
        }
    return None

def get_all_cached_videos():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, video_title, summary, analyzed_at FROM video_cache ORDER BY analyzed_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "video_id": r[0],
            "video_title": r[1],
            "summary": json.loads(r[2]),
            "analyzed_at": r[3]
        })
    return history

def log_metric(endpoint, latency, status_code):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO system_metrics (endpoint, latency, status_code, timestamp) VALUES (?, ?, ?, ?)",
        (endpoint, latency, status_code, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def log_quota(operation, units):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO quota_logs (operation, units_consumed, timestamp) VALUES (?, ?, ?)",
        (operation, units, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_aggregated_metrics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total requests
    cursor.execute("SELECT COUNT(*) FROM system_metrics")
    total_requests = cursor.fetchone()[0]
    
    # Average latency
    cursor.execute("SELECT AVG(latency) FROM system_metrics")
    avg_latency = cursor.fetchone()[0] or 0.0
    
    # Success rate
    cursor.execute("SELECT COUNT(*) FROM system_metrics WHERE status_code >= 200 AND status_code < 300")
    success_requests = cursor.fetchone()[0]
    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 100.0
    
    # Estimated YouTube API quota consumed
    cursor.execute("SELECT SUM(units_consumed) FROM quota_logs")
    quota_consumed = cursor.fetchone()[0] or 0
    
    # Latencies by endpoint
    cursor.execute("SELECT endpoint, AVG(latency), COUNT(*) FROM system_metrics GROUP BY endpoint")
    endpoint_data = [{"endpoint": r[0], "avg_latency": round(r[1], 4), "count": r[2]} for r in cursor.fetchall()]
    
    # Global sentiment distribution across all cached analyses
    cursor.execute("SELECT summary FROM video_cache")
    summaries = cursor.fetchall()
    
    global_pos = 0
    global_neu = 0
    global_neg = 0
    for s in summaries:
        try:
            s_data = json.loads(s[0])
            global_pos += s_data.get("positive", {}).get("count", 0)
            global_neu += s_data.get("neutral", {}).get("count", 0)
            global_neg += s_data.get("negative", {}).get("count", 0)
        except Exception:
            pass
        
    conn.close()
    
    return {
        "total_requests": total_requests,
        "avg_latency": round(avg_latency, 4),
        "success_rate": round(success_rate, 2),
        "quota_consumed": quota_consumed,
        "endpoints": endpoint_data,
        "global_sentiment": {
            "positive": global_pos,
            "neutral": global_neu,
            "negative": global_neg
        }
    }

def clear_all_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM video_cache")
    cursor.execute("DELETE FROM system_metrics")
    cursor.execute("DELETE FROM quota_logs")
    conn.commit()
    conn.close()
