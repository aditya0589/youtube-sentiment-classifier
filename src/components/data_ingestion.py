import pandas as pd
import sys
import googleapiclient.discovery
import json
from src.logger import logging
from src.exception import MyException
from dotenv import load_dotenv
import os

load_dotenv()

class DataIngestion:
    def __init__(self):
        self.api_key = os.getenv("YT_API_KEY")
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.api_key
        )
    
    def get_video_ids(self, channel_id: str, max_results: int = 100):
        """
        Get video IDs from a channel.
        """
        try:
            logging.info(f"Getting video IDs from channel: {channel_id}")
            request = self.youtube.search().list(
                part="id",
                channelId=channel_id,
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            video_ids = [item['id']['videoId'] for item in response['items']]
            try:
                from src.components.database import log_quota
                log_quota("get_video_ids (Search API)", 100)
            except Exception as q_err:
                logging.warning(f"Could not log search quota: {q_err}")
            logging.info(f"Got {len(video_ids)} video IDs from channel: {channel_id}")
            return video_ids
        except Exception as e:
            logging.error(f"Error getting video IDs from channel: {channel_id}")
            raise MyException(e, sys)

    def get_video_details(self, video_ids: list):
        """
        Get video details from a list of video IDs.
        """
        try:
            logging.info(f"Getting video details for {len(video_ids)} video IDs")
            request = self.youtube.videos().list(
                part="snippet, statistics",
                id=",".join(video_ids)
            )
            response = request.execute()
            try:
                from src.components.database import log_quota
                log_quota("get_video_details (Videos API)", 1)
            except Exception as q_err:
                logging.warning(f"Could not log videos quota: {q_err}")
            logging.info(f"Got {len(response['items'])} video details")
            return response['items']
        except Exception as e:
            logging.error(f"Error getting video details for {len(video_ids)} video IDs")
            raise MyException(e, sys)

    def get_comments(self, video_id: str, max_results: int = 100, page_limit: int = 3):
        """
        Get comments for a video from youtube with pagination support.
        """
        try:
            logging.info(f"Getting comments for video: {video_id} with page limit: {page_limit}")
            comments = []
            next_page_token = None
            pages_fetched = 0
            
            for page in range(page_limit):
                pages_fetched += 1
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=max_results,
                    pageToken=next_page_token
                )
                response = request.execute()
                items = response.get('items', [])
                comments.extend(items)
                logging.info(f"Page {page + 1}: Got {len(items)} comments")
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            try:
                from src.components.database import log_quota
                log_quota("get_comments (CommentThreads API)", pages_fetched)
            except Exception as q_err:
                logging.warning(f"Could not log comments quota: {q_err}")
            
            logging.info(f"Got {len(comments)} total comments for video: {video_id}")
            return comments
        except Exception as e:
            logging.error(f"Error getting comments for video: {video_id}")
            raise MyException(e, sys)

    def convert_comments_to_dataframe(self, comments: list):
        """
        Convert comments to dataframe.
        """
        try:
            logging.info(f"Converting comments to dataframe")
            df = pd.json_normalize(comments)
            df.to_csv("data/comments.csv", index=False)
            logging.info(f"Converted {len(df)} comments to dataframe")
            return df
        except Exception as e:
            logging.error(f"Error converting comments to dataframe")
            raise MyException(e, sys)

    def convert_video_details_to_dataframe(self, video_details: list):
        """
        Convert video details to dataframe.
        """
        try:
            logging.info(f"Converting video details to dataframe")
            df = pd.json_normalize(video_details)
            df.to_csv("data/video_details.csv", index=False)
            logging.info(f"Converted {len(df)} video details to dataframe")
            return df
        except Exception as e:
            logging.error(f"Error converting video details to dataframe")
            raise MyException(e, sys)

if __name__ == "__main__":
    
    ingestor = DataIngestion()
    video_ids = ingestor.get_video_ids("UC-lHJZR3Gqxm24_Vd_AJ5Yw")
    video_details = ingestor.get_video_details(video_ids)
    comments = ingestor.get_comments(video_ids[0])
    video_details_df = ingestor.convert_video_details_to_dataframe(video_details)
    comments_df = ingestor.convert_comments_to_dataframe(comments)
    print(video_details_df)
    print(comments_df)

        
    
