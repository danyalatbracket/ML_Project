import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model


class PridictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading model")
            model = load_model("artifacts/model.pkl")

            logging.info("Loading Preprocessor")
            preprocessor = load_model("artifacts/preprocessor.pkl")

            scaled_features = preprocessor.transform(features)

            logging.info("Making prediction")
            prediction = model.predict(scaled_features)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame(
                {
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score],
                }
            )
        except Exception as e:
            raise CustomException(e, sys)
