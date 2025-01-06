import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj (any): The Python object to save.

    Raises:
        CustomException: If any error occurs during the process.
    """
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring="accuracy"):
    """
    Evaluates multiple classification models using grid search for hyperparameter tuning.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target variable.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target variable.
        models (dict): Dictionary of model names and their corresponding instances.
        param (dict): Dictionary of hyperparameters for each model.
        scoring (str): Metric for evaluating models (default: "accuracy").

    Returns:
        dict: A report containing model names and their scores on the test set.

    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}  # Dictionary to store the evaluation results

        # Iterate over each model and its corresponding hyperparameters
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Perform grid search for hyperparameter tuning
            gs = GridSearchCV(model, param[model_name], scoring=scoring, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Update the model with the best parameters found by grid search
            best_model = gs.best_estimator_
            
            # Predict on the testing set
            y_test_pred = best_model.predict(X_test)

            # Calculate the chosen metric (default: accuracy) for testing data
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Store the test score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a serialized object from a file.

    Args:
        file_path (str): Path to the file containing the serialized object.

    Returns:
        object: The deserialized object.

    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        # Open the file in binary read mode and load the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)