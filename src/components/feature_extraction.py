import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import MyException

class FeatureExtraction:
    def __init__(self):
        pass

    def extract_features(self, input_file: str):
        """
        Read the labeled comments, extract only the 'text' and 'sentiment' (as 'label') columns,
        and save them for machine learning.
        """
        try:
            logging.info(f"Loading labeled comments from {input_file}")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Labeled comments file not found at {input_file}")
                
            df = pd.read_csv(input_file)
            
            logging.info("Extracting text and label columns")
            if 'text' not in df.columns or 'sentiment' not in df.columns:
                raise KeyError("Input dataframe must contain 'text' and 'sentiment' columns")
                
            df_ml = df[['text', 'sentiment']].copy()
            df_ml.rename(columns={'sentiment': 'label'}, inplace=True)
            
            df_ml.dropna(subset=['text', 'label'], inplace=True)
            
            df_ml.to_csv("data/final_features.csv", index=False)
            logging.info(f"Successfully saved feature extracted dataset to data/final_features.csv")
            
            return df_ml
        except Exception as e:
            logging.error("Error during feature extraction")
            raise MyException(e, sys)

if __name__ == "__main__":
    try:
        extractor = FeatureExtraction()
        df_ml = extractor.extract_features('data/labeled_comments.csv')
        print("Feature Extraction successful! First few records:")
        print(df_ml.head())
    except Exception as e:
        print(f"Feature Extraction failed with error: {e}")
