import sys
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.logger import logging
from src.exception import MyException

class DataTransformation:
    def __init__(self):
        try:
            if os.getenv("VERCEL") == "1":
                nltk.data.path.append("/tmp/nltk_data")
                nltk.download('vader_lexicon', download_dir="/tmp/nltk_data", quiet=True)
            else:
                nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logging.error("Error during DataTransformation initialization")
            raise MyException(e, sys)

    def get_vader_sentiment(self, text):
        """
        Determine the sentiment of text using VADER.
        """
        try:
            if pd.isna(text) or str(text).strip() == "":
                return 'neutral'
            
            scores = self.sia.polarity_scores(str(text))
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Error calculating sentiment for text: {text}")
            raise MyException(e, sys)

    def transform_data(self, file_path: str):
        """
        Read cleaned comments, perform feature renaming/selection, and label them with VADER sentiment.
        """
        try:
            logging.info(f"Loading cleaned comments from {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Cleaned comments file not found at {file_path}")
                
            df = pd.read_csv(file_path)
            
            logging.info("Selecting and renaming columns for feature engineering")
            required_cols = {
                'snippet.topLevelComment.snippet.textOriginal': 'text',
                'snippet.topLevelComment.snippet.likeCount': 'likes',
                'snippet.topLevelComment.id': 'comment_id'
            }
            
            existing_cols = [col for col in required_cols.keys() if col in df.columns]
            df_features = df[existing_cols].copy()
            
            df_features.rename(columns={col: required_cols[col] for col in existing_cols if col in required_cols}, inplace=True)
        
            logging.info("Labeling sentiments using VADER")
            if 'text' in df_features.columns:
                df_features['sentiment'] = df_features['text'].apply(self.get_vader_sentiment)
            else:
                logging.warning("No 'text' column found after renaming. Cannot compute sentiment.")
            
            output_path = 'data/labeled_comments.csv'
            df_features.to_csv(output_path, index=False)
            logging.info(f"Successfully transformed and saved labeled comments to {output_path}")
            
            return df_features
        except Exception as e:
            logging.error("Error during data transformation")
            raise MyException(e, sys)

if __name__ == "__main__":
    try:
        transformer = DataTransformation()
        df_labeled = transformer.transform_data('data/cleaned_comments.csv')
        print("Data Transformation successful! First few records:")
        print(df_labeled.head())
    except Exception as e:
        print(f"Data Transformation failed with error: {e}")
