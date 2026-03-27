from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data, test_data
        )

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()