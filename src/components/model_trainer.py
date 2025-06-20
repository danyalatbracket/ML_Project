import os
from sklearn.linear_model import LinearRegression
import sys
from catboost import CatBoostRegressor

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Instructor said also try with hyperparameter tuning
            logging.info("Initiating model training")

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
            }
            # models = {
            #     "Linear Regression": LinearRegression(),  # No major hyperparameters
            #     "K-Neighbors Regressor": KNeighborsRegressor(
            #         n_neighbors=5, weights="distance"
            #     ),
            #     "Decision Tree": DecisionTreeRegressor(
            #         max_depth=10, min_samples_split=5
            #     ),
            #     "Random Forest Regressor": RandomForestRegressor(
            #         n_estimators=100, max_depth=15, random_state=42
            #     ),
            #     "XGBRegressor": XGBRegressor(
            #         n_estimators=100,
            #         learning_rate=0.1,
            #         max_depth=6,
            #         random_state=42,
            #         verbosity=0,
            #     ),
            #     "CatBoosting Regressor": CatBoostRegressor(
            #         iterations=100, learning_rate=0.1, depth=6, verbose=False
            #     ),
            #     "AdaBoost Regressor": AdaBoostRegressor(
            #         n_estimators=100, learning_rate=0.1, random_state=42
            #     ),
            #     "Gradient Boosting Regressor": GradientBoostingRegressor(
            #         n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            #     ),
            # }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with sufficient accuracy", sys
                )
            logging.info(
                f"Best model found: {best_model_name} with score: {best_model_score}"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted_X_test = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted_X_test)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
