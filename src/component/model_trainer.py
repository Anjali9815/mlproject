# import all required libraries
import os, sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import Customexception
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModeltrainingConfig():
    trained_model_path = os.path.join("artifact", "model.pkl")


class ModelTrainer():
    def __init__(self):
        self.modeltrainer_config = ModeltrainingConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("model training started")
            x_train, y_train, x_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                    }
            
            model_report = evaluate_model(x_train, y_train, x_test, y_test, models)

            # to get best model
            best_model_score = max(sorted(model_report.values()))

            # get best model name 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            # print(best_model_name)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Customexception("NO best model Found")

            logging.info("Model training Done")

            save_object(
                file_path = self.modeltrainer_config.trained_model_path,
                obj = best_model,

            )

            predicted_op = best_model.predict(x_test)

            score_r2 = r2_score(y_test, predicted_op)

            return score_r2

        except Exception as e:
            logging.info(e)
            raise Customexception(e, sys)





