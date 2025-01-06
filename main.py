import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

def main():
    try:
        # Initialize logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logging.info("Starting the end-to-end machine learning pipeline.")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train Data: {train_data_path}, Test Data: {test_data_path}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training and Evaluation
        model_trainer = ModelTrainer()
        test_accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training and Evaluation completed. Test Accuracy: {test_accuracy}")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()
