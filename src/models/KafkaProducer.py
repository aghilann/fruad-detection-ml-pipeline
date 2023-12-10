from kafka import KafkaProducer
import csv
import json

# Kafka topic to produce to
topic = 'credit_card_transactions'  # Replace with your desired topic name

# Create Kafka producer instance with your provided configuration
producer = KafkaProducer(
    bootstrap_servers=['superb-goldfish-10068-us1-kafka.upstash.io:9092'],
    sasl_mechanism='SCRAM-SHA-256',
    security_protocol='SASL_SSL',
    sasl_plain_username='c3VwZXJiLWdvbGRmaXNoLTEwMDY4JLeFgjhjHS6GjGVu9PXkdWb2zKS_1XyxVik',
    sasl_plain_password=''  # Replace with your actual password
)

# Check if the producer is connected
# if producer.bootstrap_connected():
#     print("Kafka producer is connected to the broker")
# else:
#     print("Kafka producer is not connected to the broker")


def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')


def produce_transactions(producer_name, topic_name, data_file):
    with open(data_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for i, row in enumerate(csv_reader):
            if i >= 10:
                break
            # Exclude 'is_fraud' field
            transaction_data = {key: value for key, value in row.items() if key != 'is_fraud'}
            try:
                producer_name.send(topic=topic_name, value=json.dumps(transaction_data).encode('utf-8'))
                delivery_report(None, transaction_data)
            except Exception as e:
                delivery_report(e, None)

            break


if __name__ == '__main__':
    # Path to your CSV file (data/train_data.csv)
    data_file_path = './data/test_data.csv'
    # Produce transactions from the CSV file to the specified Kafka topic
    produce_transactions(producer, topic, data_file_path)

    # Close the producer
    producer.flush()
    producer.close()
