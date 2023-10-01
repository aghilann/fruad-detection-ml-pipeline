import os
from flytekit import task, workflow
from preprocess import preprocess_data, split_data
from spark_training import train_model
from model_evaluation import is_fraudulent_transaction
from typing import Dict
import logging

logging.basicConfig(filename='logs/workflow.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

sample_fraud_transaction = {
    "merchant": 415.0,  # Use float literals with decimal points
    "category": 4.0,
    "amt": 289.8,
    "city": 122.0,
    "state": 2.0,
    "lat": 34.298,
    "long": -114.156,
    "city_pop": 126.0,
    "job": 92.0,
    "merch_lat": 34.11963,
    "merch_long": -114.068992,
    "is_fraud": 1.0,  # Use float literals for numeric values
    "age": 30.0
}

sample_non_fraud_transaction = {
    "merchant": 123.0,
    "category": 2.0,
    "amt": 50.0,
    "city": 101.0,
    "state": 5.0,
    "lat": 35.750,
    "long": -97.556,
    "city_pop": 1000.0,
    "job": 50.0,
    "merch_lat": 35.755,
    "merch_long": -97.550,
    "is_fraud": 0.0,
    "age": 35.0
}

@workflow
def credit_card_fraud_detection_workflow(input_data_path: str, sample_fraud_transaction: Dict[str, float]) -> bool:
    preprocess_task = preprocess_data(filepath=input_data_path)

	# Define a task that splits the data into training and testing sets
    train_data, test_data = split_data(filepath=preprocess_task)
    train_task = "data/spark_data"
    if not os.path.exists(train_task):
          train_task = train_model(train_data_path=train_data)

    is_fraudulent = is_fraudulent_transaction(input_data=sample_fraud_transaction, model_path=train_task)

    return is_fraudulent

if __name__ == "__main__":
    input_data_path = "data/credit_card_fraud.csv"
    result = credit_card_fraud_detection_workflow(
        input_data_path=input_data_path,
        sample_fraud_transaction=sample_fraud_transaction
	)
    print(f'Is Fraudulent Transaction: {result}')

    
    