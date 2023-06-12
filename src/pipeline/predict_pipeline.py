# import all required library
import pandas as pd
import numpy as np
from src.exception import Customexception
from src.logger import logging
import sys, os
from src.utils import load_object

class PredictPipeline():
    def __init__(self):
        pass

    def predict1(self, features):
        try:
            model_path = 'artifact\model.pkl'
            prepprocessor_path = "artifact\preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(prepprocessor_path)

            data_scale = preprocessor.tranform(features)
            preds = model.predict(data_scale)
            
            return preds
        except Exception as e:
            raise Customexception(e,sys)
    

    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise Customexception(e,sys)


class CustomData():
    def __init__(self, gender : str, 
                 race_ethnicity : str, 
                 parental_level_of_education : str,
                 lunch : str,
                 test_preparation_course :str,
                 reading_score: int,
                 writing_score: int):
        try:
        
            self.gender = gender
            # self.race_ethncity = race_ethncity
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        except Exception as e:
            raise Customexception(e, sys)



    def get_df(self):
        try:
            custom_dict = {
                'gender' : [self.gender],
                'race_ethnicity' : [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch' : [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_dict)
        except Exception as e:
            raise Customexception(e, sys)




