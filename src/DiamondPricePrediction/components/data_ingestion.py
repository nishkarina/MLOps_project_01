import pandas as pd
import numpy as np
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        
        try:
            data = pd.read_csv(Path(os.path.join("notebooks/data", "gemstone.csv")))
            logging.info("Dataset read as a DataFrame")
            
            raw_data_dir = os.path.dirname(self.ingestion_config.raw_data_path)
            logging.info(f"Raw data directory: {raw_data_dir}")
            
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Directory created if not exists: {raw_data_dir}")
            
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset saved in artifact folder")
            
            logging.info("Performing train-test split")
            
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exception occurred during data ingestion stage")
            raise customexception(e, sys)

# Example usage
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print(f"Train data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")
