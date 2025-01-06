import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the Model Trainer.
    Defines the path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Class to handle model training, evaluation, and selection.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple classification models, evaluates their performance, and saves the best one.

        Args:
            train_array (numpy.ndarray): Array containing training data (features and target).
            test_array (numpy.ndarray): Array containing testing data (features and target).

        Returns:
            float: Accuracy score of the best model on the testing dataset.

        Raises:
            CustomException: If no model performs well or any error occurs during the process.
        """
        try:
            logging.info("Splitting training and testing data into features and target variables.")
            # Splitting train and test data into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Dictionary of classification models to evaluate
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "XGBClassifier": XGBClassifier(random_state=42, eval_metric='logloss'),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
            }


            # Dictionary of hyperparameters for each model
            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "lbfgs"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
                "Support Vector Classifier": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                },
                "CatBoostClassifier": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "iterations": [50, 100, 200],
                },
            }

            logging.info("Evaluating models with cross-validation and hyperparameter tuning.")
            # Evaluate models and store their accuracy scores
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params, scoring="accuracy"
            )

            # Get the best model's score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Check if the best model meets the minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with satisfactory performance.")

            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")

            # Fit the best model on the training data
            logging.info("Fitting the best model on the training dataset.")
            best_model.fit(X_train, y_train)  # Fit the model here

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the testing dataset with the best model
            predicted = best_model.predict(X_test)  # Now the model is fitted, so this works

            # Calculate the accuracy score for the testing dataset
            accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Test accuracy: {accuracy}")

            return accuracy


        except Exception as e:
            raise CustomException(e, sys)
