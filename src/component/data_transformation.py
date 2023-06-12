# feature engineering
import sys, os
from dataclasses import dataclass
import numpy as np
from src.exception import Customexception
from src.logger import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass

class DataTransformconfig():
    preprocessor_ob_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation():

    def __init__(self):
        self.data_transformation_config = DataTransformconfig()

    def get_data_transformer_object(self):
        try:
            numerical_data = ['writing_score', 'reading_score']
            categorical_data = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # create numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "median")), # handling median
                    ("scaler", StandardScaler()) # handling categorical data

                ])
            
            categorical_pipeline = Pipeline(
                    steps=[
                    ("imputer", SimpleImputer(strategy= "most_frequent")), # handling median
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean = False)) # handling categorical data

                ])
            logging.info("categorical column encoding completed")
            logging.info("numerical column encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_data),
                ("categorical_columns", categorical_pipeline, categorical_data)]
            )

            return preprocessor
        except Exception as e:
            logging.info(e)
            raise Customexception(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("read train and test data")
            train_Df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("obtaining the preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_column = ["reading_score", "writing_score"]

            input_feature_train_df = train_Df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_Df[target_column]

            input_feature_test_df = test_df.drop(columns = target_column, axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.preprocessor_ob_file_path, obj= preprocessor_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path)
        except Exception as e:
            raise Customexception(e, sys)
