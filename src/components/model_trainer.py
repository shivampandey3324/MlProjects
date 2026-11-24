import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path,):
        try:
            logging.info("Splitting the train and test array")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )


            models={
                "RandomRegressor":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "LinearRegression":LinearRegression(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor()
            }

            params = {
    "RandomRegressor": {
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "DecisionTreeRegressor": {
        'criterion': ['squared_error', 'absolute_error'],
        # 'splitter': ['best', 'random'],
        # 'max_features': ['sqrt', 'log2'],
    },

    "LinearRegression": {},

    "AdaBoostRegressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        # 'loss': ['linear', 'square', 'exponential'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "GradientBoostingRegressor": {
        # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # 'criterion': ['squared_error', 'friedman_mse'],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "CatBoostRegressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },

    "KNeighborsRegressor": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },

    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
                    }

            
            model_report:dict=evaluate_models(x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test,
                                  models=models,params=params)


            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("None are efficient")

            logging.info("Best model found")   

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2score=r2_score(y_test,predicted)

            return (r2score,best_model_name)


        
        except Exception as e:
            raise CustomException(e,sys)
            