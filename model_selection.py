import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_selection(file_path):
    # Load the preprocessed data
    data = pd.read_csv(file_path)
    
    # Split the data into features and target
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TPOTClassifier with fewer generations and population size for a quicker run
    tpot = TPOTClassifier(generations=2, population_size=10, verbosity=2, random_state=42, n_jobs=-1)
    
    # Run TPOT
    tpot.fit(X_train, y_train)

    # Print the best pipeline
    logger.info("Best Pipeline: %s", tpot.fitted_pipeline_)

    # Score the best pipeline on the test data
    score = tpot.score(X_test, y_test)
    logger.info("Best Pipeline Score: %s", score)

    # Export the script for the best pipeline
    tpot.export('best_pipeline.py')

if __name__ == "__main__":
    input_file = 'data/processed_data.csv'  # replace with the path to your processed data file
    model_selection(input_file)
