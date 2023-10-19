import os
import logging
import pandas as pd
from typing import Dict
from io import StringIO
import tempfile
import tarfile
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from src.models.S3 import S3
from flytekit import task

class ModelEvaluator:

    def __init__(self, s3_bucket_name):
        self.s3 = S3(bucket_name=s3_bucket_name)

    def download_and_extract_tarball(self, folder_path: str, file_name: str, destination: str) -> None:
        binary_content = self.s3.read_binary(folder_path, file_name)
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(binary_content)
            tmp_file.flush()
            with tarfile.open(tmp_file.name, "r:gz") as tar:
                tar.extractall(path=destination)

    @staticmethod
    def setup_logging():
        filename = os.path.splitext(os.path.basename(__file__))[0]
        logging.basicConfig(filename=f'logs/{filename}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def evaluate_model(self, model_path: str, test_data_path: str, model_name: str, input_cols: list) -> None:
        self.setup_logging()

        spark = SparkSession.builder.appName("FraudDetectionModelEvaluation").getOrCreate()

        temp_dir = tempfile.mkdtemp()
        self.download_and_extract_tarball(model_path, model_name, temp_dir)
        extracted_model_path = os.path.join(temp_dir, "model")

        logging.info("Loading the saved model")
        model = RandomForestClassificationModel.load(extracted_model_path)

        raw_csv = self.s3.read_file(model_path, test_data_path)
        test_df = spark.createDataFrame(pd.read_csv(StringIO(raw_csv)))

        vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        test_df = vector_assembler.transform(test_df)

        transformed_test_df = model.transform(test_df)
        label_indexer = StringIndexer(inputCol='is_fraud', outputCol='label')
        transformed_test_df = label_indexer.fit(transformed_test_df).transform(transformed_test_df)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

        metrics = {
            "accuracy": evaluator.evaluate(transformed_test_df, {evaluator.metricName: "accuracy"}),
            "precision": evaluator.evaluate(transformed_test_df, {evaluator.metricName: "weightedPrecision"}),
            "recall": evaluator.evaluate(transformed_test_df, {evaluator.metricName: "weightedRecall"}),
            "f1": evaluator.evaluate(transformed_test_df, {evaluator.metricName: "f1"})
        }

        for metric_name, metric_value in metrics.items():
            logging.info(f'Model {metric_name.capitalize()}: {metric_value}')

        spark.stop()
        return metrics

    def is_fraudulent_transaction(self, input_data: Dict[str, float], model_path: str, input_cols: list) -> bool:
        spark = SparkSession.builder.appName("FraudDetectionModelEvaluation").getOrCreate()
        try:
            model = RandomForestClassificationModel.load(model_path)
            input_df = pd.DataFrame([input_data])
            input_df = spark.createDataFrame(input_df)
            vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
            input_df = vector_assembler.transform(input_df)
            predictions = model.transform(input_df)
            prediction = predictions.select('prediction').collect()[0]['prediction']
            return int(prediction) == 1
        except Exception as e:
            logging.error(f'Error occurred: {str(e)}')
            return False
        finally:
            spark.stop()


if __name__ == "__main__":
    INPUT_COLS = ['merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long', 'age']
    evaluator = ModelEvaluator(s3_bucket_name='mlinfra')
    evaluator.evaluate_model(model_path="data", 
                             test_data_path="test_data.csv", 
                             model_name="credit_card_fraud_model.tar.gz",
                             input_cols=INPUT_COLS)
