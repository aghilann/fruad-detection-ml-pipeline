import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from flytekit import task

def setup_logging():
    logging.basicConfig(filename='logs/spark_model_training.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

@task
def train_model(train_data_path: str) -> str:
    logger = setup_logging()
    logger.info('Start Spark Application')
    model_path = None
    try:
        spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()
        logger.info('Created Spark Session')
        
        logger.info(f'Reading training data from {train_data_path}')
        train_df = spark.read.csv(train_data_path, header=True, inferSchema=True)
        
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
        
        model_path = "data/spark_data"
        model.save(model_path)
        logger.info(f'Model saved to {model_path}')
        
    except Exception as e:
        logger.error(f'Error occurred: {str(e)}', exc_info=True)
        raise
    finally:
        spark.stop()
        logger.info('Stopped Spark Session')
        return model_path

if __name__ == "__main__":
    train_model("data/train_data.csv")
