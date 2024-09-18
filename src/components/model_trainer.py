import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_fil_path = os.path.join('artifacts',"model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_Trainer(self, train_arrray, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test=(
                train_arrray[:,:-1],
                train_arrray[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            # To find the best model score from dict
            best_model_score= max(sorted(model_report.values()))
            # To find the best model name from best model score from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info("Best model is {0}".format(best_model_name))

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on training and test data set")

            save_object(
                file_path= self.model_trainer_config.trained_model_fil_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2score= r2_score(y_test, predicted)

            return r2score


        except Exception as e:
            raise CustomException(e,sys)        