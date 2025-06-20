import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as _file:
            dill.dump(obj, _file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        from src.logger import logging

        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            y_train_r2_square = r2_score(y_test, y_test_pred)

            y_test_r2_square = r2_score(y_test, y_test_pred)

            model_report[model_name] = y_test_r2_square

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
