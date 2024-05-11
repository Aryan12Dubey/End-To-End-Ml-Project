import os
import sys
from src.exception import CustomException
import pandas as pd
from src.logger import logging
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    row_data_path: str=os.path.join('artifacts',"data.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
         df=pd.read_csv('data/StudentsPerformance.csv')
         logging.info('Reading the data ')
         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.row_data_path,index=False,header=True)

         logging.info('train Test seperation started')

         train,test=train_test_split(df,test_size=0.2)

         train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

         test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

         logging.info('Seperation of data completed')
        
         return(
           self.ingestion_config.train_data_path,
           self.ingestion_config.test_data_path
         )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transform(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))