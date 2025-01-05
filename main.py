import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.model_saver import save_model

def main():
    try:
        logging.info("Starting the end-to-end machine learning pipeline.")
        
        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train Data: {train_data_path}, Test Data: {test_data_path}")
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation completed.")
        
        # Step 3: Model Training and Evaluation
        model_trainer = ModelTrainer()
        trained_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model Training and Evaluation completed.")
        
        # Step 4: Save the Trained Model
        save_model(trained_model)
        logging.info("Model saved successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")
        raise

if __name__ == "__main__":
    # Run the main pipeline
    main()
