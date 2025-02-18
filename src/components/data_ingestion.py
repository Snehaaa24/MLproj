import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split #bcoz in order to ingest data you have to split it
from dataclasses import dataclass

class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train.csv")
    test_data_path=os.path.join('artifacts',"test.csv")
    raw_data_path=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.injestion_config=DataIngestionConfig()
    def initiate_data_injestion(self):
        logging.info("Data injestion has been initiated.")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("The stud.csv file is being read as df ")

            #creates the folder 'artifacts'
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)

            #creates the raw data file
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split has been initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=24) 
            train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True) #creates the train data file
            test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True) #creates the test data file

            logging.info("Data ingestion was successful.")

            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_injestion()