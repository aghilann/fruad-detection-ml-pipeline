import logging
import pandas as pd
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

app = Flask(__name__)

def setup_logging():
    logging.basicConfig(filename='logs/spark_model_prediction.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def predict_fraud(json_data, model_path):
    logger = setup_logging()
    logger.info('Start Spark Application')

    try:
        spark = SparkSession.builder.appName("FraudDetectionModelPrediction").getOrCreate()
        logger.info('Created Spark Session')

        # Load the trained model
        model = RandomForestClassificationModel.load(model_path)
        logger.info(f'Loaded model from {model_path}')

        # Prepare the input data
        input_data = pd.DataFrame([json_data])
        input_data = spark.createDataFrame(input_data)

        # Vectorize the input data
        input_cols = ['merchant', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long', 'age']
        vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        input_data = vector_assembler.transform(input_data)

        # Make predictions
        predictions = model.transform(input_data)
        if predictions.count() == 0:
            logger.error('No predictions was made')
            return {'error': 'No predictions was made'}    
        prediction = predictions.select('prediction').collect()[0]['prediction']
        result = {
            'is_fraud': int(prediction)
        }

        return result

    except Exception as e:
        logger.error(f'Error occurred: {str(e)}', exc_info=True)
        return {'error': 'An error occurred'}

    finally:
        spark.stop()
        logger.info('Stopped Spark Session')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_path = "data/spark_data"
    prediction_result = predict_fraud(data, model_path)
    return jsonify(prediction_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

