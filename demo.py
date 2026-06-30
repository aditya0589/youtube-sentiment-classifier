import sys
import os
import re
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.model_predictor import ModelPredictor

def extract_video_id(url: str) -> str:
    """
    Extract the 11-character video ID from a standard YouTube URL.
    """
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return url.strip() # Return directly if they provided just the raw video ID

def run_demo():
    print("==================================================")
    print("   YOUTUBE COMMENT SENTIMENT ANALYSIS DEMO        ")
    print("==================================================")
    
    # 1. Get YouTube URL or Video ID
    url_input = input("Enter YouTube Video URL or Video ID: ").strip()
    if not url_input:
        print("Error: No input provided. Exiting.")
        return
        
    video_id = extract_video_id(url_input)
    print(f"\n[1/4] Extracted Video ID: {video_id}")
    
    # 2. Ingest comments in real-time
    print("[2/4] Fetching comments from YouTube API...")
    try:
        ingestor = DataIngestion()
        # Fetching up to 100 comments from the video
        comments = ingestor.get_comments(video_id=video_id, max_results=100, page_limit=1)
        if not comments:
            print("Error: No comments found or comments are disabled on this video.")
            return
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return
        
    # 3. Clean comments
    print("[3/4] Cleaning comments (filtering English and formatting)...")
    cleaning = DataCleaning()
    df_comments = ingestor.convert_comments_to_dataframe(comments)
    df_clean = cleaning.clean_data(df_comments)
    df_clean = cleaning.remove_non_english(df_clean)
    df_clean = cleaning.remove_symbols(df_clean)
    
    # Extract the clean text column
    text_column = 'snippet.topLevelComment.snippet.textOriginal'
    if text_column not in df_clean.columns or len(df_clean) == 0:
        print("Error: No clean English comments left to classify.")
        return
        
    texts = df_clean[text_column].tolist()
    print(f"Successfully cleaned and prepared {len(texts)} comments.")
    
    # 4. Predict sentiment
    print("[4/4] Running sentiment predictions...")
    try:
        predictor = ModelPredictor()
        results = []
        for text in texts:
            sentiment = predictor.predict(text)
            results.append((text, sentiment))
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return
        
    # 5. Display Sentiment Analysis Report
    df_results = pd.DataFrame(results, columns=["text", "sentiment"])
    counts = df_results["sentiment"].value_counts()
    total = len(df_results)
    
    pos_pct = (counts.get("positive", 0) / total) * 100
    neu_pct = (counts.get("neutral", 0) / total) * 100
    neg_pct = (counts.get("negative", 0) / total) * 100
    
    print("\n" + "="*50)
    print("             SENTIMENT ANALYSIS REPORT            ")
    print("="*50)
    print(f"Total English Comments Analyzed: {total}")
    print(f"Positive Sentiment:  {pos_pct:.2f}% ({counts.get('positive', 0)})")
    print(f"Neutral Sentiment:   {neu_pct:.2f}% ({counts.get('neutral', 0)})")
    print(f"Negative Sentiment:  {neg_pct:.2f}% ({counts.get('negative', 0)})")
    print("="*50)
    
    # Print some examples for each class
    for sentiment_type in ["positive", "neutral", "negative"]:
        subset = df_results[df_results["sentiment"] == sentiment_type].head(3)
        if len(subset) > 0:
            print(f"\nExamples of {sentiment_type.upper()} comments:")
            for idx, row in subset.iterrows():
                # Truncate comment if it's very long
                truncated_text = row['text'][:90] + "..." if len(row['text']) > 90 else row['text']
                print(f" - \"{truncated_text}\"")

if __name__ == "__main__":
    run_demo()