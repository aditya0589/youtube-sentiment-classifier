from src.logger import logging
from src.exception import MyException
import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory
import sys
import os


class DataCleaning:
    def __init__(self):
        pass

    def clean_data(self, df):
        """
        Clean the dataframe.
        """
        try:
            logging.info("Cleaning the dataframe")
            df = df.drop_duplicates()
            df = df.dropna()
            logging.info(f"Cleaned {len(df)} rows from the dataframe")
            return df
        except Exception as e:
            logging.error("Error cleaning the dataframe")
            raise MyException(e, sys)

    def remove_non_english(self, df):
        """
        Remove non-english comments to make the language processing easier
        """
        try:
            logging.info("Removing non-english comments")
            DetectorFactory.seed = 0
            
            def is_english(text):
                if not isinstance(text, str) or not text.strip():
                    return False
                try:
                    return detect(text) == 'en'
                except LangDetectException:
                    return False
            
            df = df[df['snippet.topLevelComment.snippet.textOriginal'].apply(is_english)]
            logging.info(f"Cleaned {len(df)} rows from the dataframe")
            try:
                os.makedirs("data", exist_ok=True)
                df.to_csv('data/cleaned_comments.csv', index=False)
            except Exception as write_err:
                logging.warning(f"Could not save cleaned comments CSV: {write_err}. Continuing in-memory.")
            return df
        except Exception as e:
            logging.error("Error removing non-english comments")
            raise MyException(e, sys)

    def remove_symbols(self, df):
        """
        Remove symbols from the dataframe.
        """
        try:
            logging.info("Removing symbols from the dataframe")
            df['snippet.topLevelComment.snippet.textOriginal'] = df['snippet.topLevelComment.snippet.textOriginal'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
            logging.info(f"Cleaned {len(df)} rows from the dataframe")
            try:
                os.makedirs("data", exist_ok=True)
                df.to_csv('data/cleaned_comments.csv', index=False)
            except Exception as write_err:
                logging.warning(f"Could not save cleaned comments CSV: {write_err}. Continuing in-memory.")
            return df
        except Exception as e:
            logging.error("Error removing symbols from the dataframe")
            raise MyException(e, sys)


if __name__ == "__main__":
    cleaning = DataCleaning()
    df = pd.read_csv('data/comments.csv')
    df = cleaning.clean_data(df)
    df = cleaning.remove_non_english(df)
    df = cleaning.remove_symbols(df)
    print(df)