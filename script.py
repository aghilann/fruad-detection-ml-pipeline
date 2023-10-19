import pandas as pd

# Read the CSV file
df = pd.read_csv('data/credit_card_fraud.csv')

# Duplicate the rows
df_duplicated = pd.concat([df] * 20, ignore_index=True)

# Save the duplicated dataframe to a new CSV file
df_duplicated.to_csv('data/credit_card_fraud_big.csv', index=False)