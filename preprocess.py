import pandas as pd
import logging
from flytekit import task, workflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from io import StringIO
from src.models.S3 import S3

# Configure Logging
logging.basicConfig(filename='logs/preprocess.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

s3 = S3(bucket_name='mlinfra')

@task
def preprocess_data(folder_path: str, file_name: str) -> str:
    logging.info('Starting Data Preprocessing')
    
    try:
        # Load Data
        file_content = s3.read_file(folder_path, file_name)
        data = pd.read_csv(StringIO(file_content))
        
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
        processed_file_name = 'processed_data.csv'
        logging.info(f'Saving processed data to {processed_file_name}')
        s3.write_file(folder_path, processed_file_name, data.to_csv(index=False))
        return processed_file_name
        
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise

@task
def split_data(folder_path: str, file_name: str) -> Tuple[str, str]:
    logging.info('Starting to split Data')
    
    try:
        # Load the preprocessed data
        logging.info(f'Loading processed data from {file_name}')
        file_content = s3.read_file(folder_path, file_name)
        data = pd.read_csv(StringIO(file_content))
        
        # Split the data into training and testing sets and save to new files
        logging.info('Splitting Data into Train and Test')
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        train_data_file_name = 'train_data.csv'
        test_data_file_name = 'test_data.csv'
        
		# Save the the training and testing data to S3
        s3.write_file(folder_path, train_data_file_name, train_data.to_csv(index=False))
        s3.write_file(folder_path, test_data_file_name, test_data.to_csv(index=False))
        
        return train_data_file_name, test_data_file_name
		
        
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise


if __name__ == "__main__":
    processed_file_name = preprocess_data(
        folder_path='data',
        file_name='credit_card_fraud.csv'
    )
    print(split_data(folder_path='data', file_name=processed_file_name))
