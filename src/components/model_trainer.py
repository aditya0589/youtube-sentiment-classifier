import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import mlflow
import dagshub
from src.logger import logging
from src.exception import MyException

class ModelTrainer:
    def __init__(self):
        try:
            self.artifacts_dir = "artifacts"
            os.makedirs(self.artifacts_dir, exist_ok=True)
            self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
            self.vectorizer_path = os.path.join(self.artifacts_dir, "vectorizer.pkl")
            self.matrix_path = os.path.join(self.artifacts_dir, "confusion_matrix.png")
            
            logging.info("Initializing DagsHub MLflow tracking integration")
            dagshub.init(
                repo_owner='aditya0589', 
                repo_name='youtube-sentiment-classifier', 
                mlflow=True
            )
        except Exception as e:
            logging.error("Initialization of ModelTrainer failed")
            raise MyException(e, sys)

    def train_model(self, data_path: str = "data/final_features.csv"):
        """
        Train the optimal Logistic Regression model on TF-IDF features and log artifacts to MLflow.
        """
        try:
            logging.info(f"Loading final features from {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")

            df = pd.read_csv(data_path)
            
            logging.info("Splitting dataset into train and test sets")
            X = df['text'].astype(str)
            y = df['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            train_df = pd.DataFrame({'text': X_train, 'label': y_train})
            test_df = pd.DataFrame({'text': X_test, 'label': y_test})
            train_csv = os.path.join("data", "train.csv")
            test_csv = os.path.join("data", "test.csv")
            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)

            logging.info("Vectorizing text using TF-IDF")
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), stop_words=None)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            logging.info("Training best Logistic Regression model")
            model = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
            model.fit(X_train_vec, y_train)

            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logging.info(f"Model evaluation metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            logging.info("Generating confusion matrix plot")
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            plt.title("Confusion Matrix - Best Logistic Regression Model")
            plt.savefig(self.matrix_path)
            plt.close()

            logging.info("Saving vectorizer and model artifacts locally")
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)

            logging.info("Logging run details to MLflow")
            mlflow.set_experiment('yt-comment-classifier-experimenta')
            with mlflow.start_run(run_name="production_logistic_regression"):
                mlflow.log_param("max_features", 1000)
                mlflow.log_param("ngram_range", "(1, 1)")
                mlflow.log_param("stop_words", "None")
                mlflow.log_param("C", 1.0)
                mlflow.log_param("class_weight", "balanced")
                
   
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                

                mlflow.log_artifact(train_csv)
                mlflow.log_artifact(test_csv)
                mlflow.log_artifact(self.matrix_path)
                mlflow.log_artifact(self.vectorizer_path)
                mlflow.log_artifact(self.model_path)
                
            logging.info("Model training and logging successfully completed")
            return accuracy, f1

        except Exception as e:
            logging.error("Error during model training execution")
            raise MyException(e, sys)

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.train_model()
        print("Model training pipeline completed successfully!")
    except Exception as e:
        print(f"Model training pipeline failed with error: {e}")
