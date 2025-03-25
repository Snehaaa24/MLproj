import os
import sys
from src.utils import evaluate_model, save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting the train and test input data.")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1], # all rows, all cols except last col (1)
                train_arr[:,-1], # all rows, only last col (2)
                test_arr[:,:-1],# (1)
                test_arr[:,-1] # (2)
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score=max(model_report.values()) #To find the best model score

            best_model_name = max(model_report, key=model_report.get) #to get best model name

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found.")
            
            logging.info("Best model found.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square= r2_score(y_test,predicted)
            logging.info(f"The best performing model was ->{best_model_name}<- with r2_score of:{r2_square}")
            return r2_square,best_model_name

        except Exception as e:
            raise CustomException(e,sys)