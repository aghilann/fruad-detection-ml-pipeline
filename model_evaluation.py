from typing import Dict
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
import logging
import pandas as pd
from flytekit import task

@task
def evaluate_model(model_path: str) -> None:
    # Set up logging
    logging.basicConfig(filename='logs/model_evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create Spark session
    spark = SparkSession.builder.appName("FraudDetectionModelEvaluation").getOrCreate()

    logging.info("Loading the saved model")
    model = RandomForestClassificationModel.load(model_path)

    logging.info("Reading the test data")
    test_data_path = "data/test_data.csv"
    test_df = spark.read.csv(test_data_path, header=True, inferSchema=True)

    # Indexing the label column in the test data
    label_indexer = StringIndexer(inputCol='is_fraud', outputCol='label')
    test_df = label_indexer.fit(test_df).transform(test_df)
    
    # Specify the input columns for VectorAssembler in the test data
    input_cols = ['merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long', 'age']
    vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
    test_df = vector_assembler.transform(test_df)
    
    logging.info("Transforming the test data using the loaded model")
    transformed_test_df = model.transform(test_df)
    
    logging.info("Evaluating the model")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    
    accuracy = evaluator.evaluate(transformed_test_df, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(transformed_test_df, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(transformed_test_df, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(transformed_test_df, {evaluator.metricName: "f1"})

    logging.info(f'Model Accuracy: {accuracy}')
    logging.info(f'Model Precision: {precision}')
    logging.info(f'Model Recall: {recall}')
    logging.info(f'Model F1-Score: {f1}')

    print(f'Model Accuracy: {accuracy}')
    print(f'Model Precision: {precision}')
    print(f'Model Recall: {recall}')
    print(f'Model F1-Score: {f1}')
    
    # Stop the SparkSession
    spark.stop()

@task
def is_fraudulent_transaction(input_data: Dict[str, float], model_path: str) -> bool:
    # Initialize SparkSession
    spark = SparkSession.builder.appName("FraudDetectionModelEvaluation").getOrCreate()

    try:
        # Load the trained model
        model = RandomForestClassificationModel.load(model_path)

        # Prepare the input data
        input_df = pd.DataFrame([input_data])
        input_df = spark.createDataFrame(input_df)

        # Define input columns
        input_cols = ['merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long', 'age']

        # Create a vector assembler
        vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        input_df = vector_assembler.transform(input_df)

        # Make predictions
        predictions = model.transform(input_df)

        # Extract the predicted label (0 for non-fraud, 1 for fraud)
        prediction = predictions.select('prediction').collect()[0]['prediction']

        # Return True if the prediction indicates fraud (1), otherwise return False
        return int(prediction) == 1

    except Exception as e:
        print(f'Error occurred: {str(e)}')
        return False

    finally:
        # Stop the SparkSession
        spark.stop()

if __name__ == "__main__":
    model_path = "data/spark_data"
    evaluate_model(model_path=model_path)
