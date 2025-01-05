import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Defines the path to save the preprocessor object.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Handles data transformation processes, including creating preprocessing pipelines.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create a preprocessing object with pipelines for different column types.

        Returns:
            ColumnTransformer: Preprocessing object.
        """
        try:
            binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            nominal_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                            'DeviceProtection', 'TechSupport', 'PaymentMethod']
            ordinal_cols = ['Contract']
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

            # Ordinal mapping for 'Contract'
            contract_mapping = [['Month-to-month', 'One year', 'Two year']]

            # Preprocessing pipelines
            binary_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='if_binary', dtype=int))
            ])
            nominal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=contract_mapping))
            ])
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Binary columns: {binary_cols}")
            logging.info(f"Nominal columns: {nominal_cols}")
            logging.info(f"Ordinal columns: {ordinal_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('binary', binary_transformer, binary_cols),
                    ('nominal', nominal_transformer, nominal_cols),
                    ('ordinal', ordinal_transformer, ordinal_cols),
                    ('numerical', numerical_transformer, numerical_cols)
                ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation on train and test datasets.

        Args:
            train_path (str): Path to training dataset.
            test_path (str): Path to testing dataset.

        Returns:
            tuple: Transformed train and test data, and preprocessor object path.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Convert 'TotalCharges' to numeric, handling errors
            for df in [train_df, test_df]:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            logging.info("Data loading completed. Preprocessing begins.")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Churn"

            # Validate target column existence
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in datasets.", sys)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            logging.info("Preprocessing complete. Saving object.")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
