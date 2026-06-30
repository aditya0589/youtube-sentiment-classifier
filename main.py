import sys
import os
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.feature_extraction import FeatureExtraction
from src.components.model_trainer import ModelTrainer

def run_pipeline(channel_id: str = "UC-lHJZR3Gqxm24_Vd_AJ5Yw"):
    try:
        os.makedirs("data", exist_ok=True)
        
        logging.info("Starting Youtube Sentiment Classifier Pipeline execution")
        
        logging.info("--- Phase 1: Data Ingestion ---")
        ingestor = DataIngestion()
        
        video_ids = ingestor.get_video_ids(channel_id=channel_id)
        if not video_ids:
            raise ValueError(f"No videos found for channel {channel_id}")
            
        video_details = ingestor.get_video_details(video_ids)
        ingestor.convert_video_details_to_dataframe(video_details)
        
        target_videos = video_ids[:5]
        all_comments = []
        for vid in target_videos:
            try:
                comments = ingestor.get_comments(video_id=vid, max_results=100, page_limit=10)
                all_comments.extend(comments)
            except Exception as e:
                logging.warning(f"Could not retrieve comments for video {vid}: {e}")
        
        if not all_comments:
            raise ValueError(f"No comments could be retrieved for channel {channel_id}")
            
        comments_df = ingestor.convert_comments_to_dataframe(all_comments)
        
        logging.info("--- Phase 2: Data Cleaning ---")
        cleaning = DataCleaning()
        cleaned_df = cleaning.clean_data(comments_df)
        cleaned_df = cleaning.remove_non_english(cleaned_df)
        cleaned_df = cleaning.remove_symbols(cleaned_df)
        
        logging.info("--- Phase 3: Data Transformation ---")
        transformer = DataTransformation()
        transformer.transform_data(file_path='data/cleaned_comments.csv')
        
        logging.info("--- Phase 4: Feature Extraction ---")
        extractor = FeatureExtraction()
        extractor.extract_features(input_file='data/labeled_comments.csv')
        
        logging.info("Pipeline executed successfully! Final features are saved at data/final_features.csv")
        print("Pipeline execution completed successfully!")
        
    except Exception as e:
        logging.error("Pipeline execution failed")
        raise MyException(e, sys)

if __name__ == "__main__":
    run_pipeline()
