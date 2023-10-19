from io import StringIO
import logging
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from flytekit import task
from src.models.S3 import S3

def setup_logging():
    logging.basicConfig(filename='logs/spark_model_training.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

s3 = S3(bucket_name='mlinfra')

@task
def train_model(train_data_path: str) -> str:
    logger = setup_logging()
    logger.info('Start Spark Application')
    model_path = None
    try:
        spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()
        logger.info('Created Spark Session')
        
        logger.info(f'Reading training data from {train_data_path}')
        raw_csv = s3.read_file('data', train_data_path)
        train_df = spark.createDataFrame(pd.read_csv(StringIO(raw_csv)))
        
        logger.info('Displaying data format:')
        train_df.printSchema()
        
        logger.info('Displaying first 5 rows:')
        train_df.show(5)
        
        label_indexer = StringIndexer(inputCol='is_fraud', outputCol='label')
        logger.info('Configured label indexer')
        
        input_cols = ['merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long', 'age']
        vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        logger.info('Configured vector assembler')
        
        train_df = label_indexer.fit(train_df).transform(train_df)
        train_df = vector_assembler.transform(train_df)
        logger.info('Transformed data frame')

        rf_model = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100, impurity='entropy', maxDepth=10, seed=42)
        logger.info('Initialized RandomForestClassifier')

        model = rf_model.fit(train_df)
        logger.info('Model training completed')

        try:
            # Save the model to local directory
            temp_model_path = "/tmp/model"
            model.save(temp_model_path)
            logger.info(f'Model saved locally at {temp_model_path}')
            
            # Tar/gzip the directory
            import tarfile
            tar_path = "/tmp/model.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(temp_model_path, arcname=os.path.basename(temp_model_path))
            logger.info(f'Model tarred at {tar_path}')
            
            # Upload the tarball to S3
            with open(tar_path, 'rb') as data:
                s3.write_file('data', "credit_card_fraud_model" + ".tar.gz", data)
            logger.info(f'Model uploaded to S3 at data/{"credit_card_fraud_model"}.tar.gz')

        except Exception as e:
            logger.error(f'Error occurred: {str(e)}', exc_info=True)
            raise

    except Exception as e:
        logger.error(f'Error occurred: {str(e)}', exc_info=True)
        raise

    finally:
        spark.stop()
        logger.info('Stopped Spark Session')
        return train_data_path

if __name__ == "__main__":
    train_model(train_data_path="train_data.csv")
