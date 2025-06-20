import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as _file:
            dill.dump(obj, _file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        from src.logger import logging

        model_report = {}

        for model_index, (model_name, model) in enumerate(models.items()):
            logging.info(f"Training model: {model_name}")

            param = params.get(model_name, {})

            grid_search = GridSearchCV(
                estimator=model, param_grid=param, cv=3, n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            print(f"Best parameters for {model_name}: {grid_search.best_params_}")

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            y_train_r2_square = r2_score(y_train, y_train_pred)

            y_test_r2_square = r2_score(y_test, y_test_pred)

            model_report[model_name] = y_test_r2_square

        return model_report

    except Exception as e:
        raise CustomException(e, sys)


def load_model(file_path):
    try:
        with open(file_path, "rb") as _file:
            model = dill.load(_file)
        return model
    except Exception as e:
        raise CustomException(e, sys)
