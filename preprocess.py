import pandas as pd
import logging
from flytekit import task, workflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

# Configure Logging
logging.basicConfig(filename='logs/preprocess.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@task
def preprocess_data(filepath: str) -> str:
    logging.info('Starting Data Preprocessing')
    
    try:
        # Load Data
        logging.info(f'Loading data from {filepath}')
        data = pd.read_csv(filepath)
        
        # Drop unnecessary columns
        logging.info('Dropping unnecessary columns')
        data = data.drop(columns=['trans_num', 'trans_date_trans_time'])
        
        # Encode Categorical Variables
        logging.info('Encoding Categorical Variables')
        le = LabelEncoder()
        categorical_cols = ['merchant', 'category', 'city', 'state', 'job']
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])
        
        # Convert DOB to Age
        logging.info('Converting DOB to Age')
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = ((pd.Timestamp.now() - data['dob']).dt.total_seconds() / (365.25 * 24 * 3600)).astype(int)
        data = data.drop(columns=['dob'])
        
        # Save the preprocessed data to a new file and return the path
        processed_filepath = 'data/processed_data.csv'
        logging.info(f'Saving processed data to {processed_filepath}')
        data.to_csv(processed_filepath, index=False)
        return processed_filepath
        
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise

@task
def split_data(filepath: str) -> Tuple[str, str]:
    logging.info('Starting to split Data')
    
    try:
        # Load the preprocessed data
        logging.info(f'Loading processed data from {filepath}')
        data = pd.read_csv(filepath)
        
        # Split the data into training and testing sets and save to new files
        logging.info('Splitting Data into Train and Test')
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data_filepath = 'data/train_data.csv'
        test_data_filepath = 'data/test_data.csv'
        train_data.to_csv(train_data_filepath, index=False)
        test_data.to_csv(test_data_filepath, index=False)
        return train_data_filepath, test_data_filepath
		
        
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise


if __name__ == "__main__":
    processed_filepath = preprocess_data('data/credit_card_fraud.csv')
    split_data(processed_filepath)
