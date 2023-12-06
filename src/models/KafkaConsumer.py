from kafka import KafkaConsumer
import json
from Transaction import Transaction
from dotenv import load_dotenv
import os

load_dotenv()
sasl_plain_password = os.environ.get("sasl_plain_password")


class KafkaTransactionConsumer:
    consumer: KafkaConsumer

    def __init__(self, topic_name: str):
        self.consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=['superb-goldfish-10068-us1-kafka.upstash.io:9092'],
            sasl_mechanism='SCRAM-SHA-256',
            security_protocol='SASL_SSL',
            sasl_plain_username='c3VwZXJiLWdvbGRmaXNoLTEwMDY4JLeFgjhjHS6GjGVu9PXkdWb2zKS_1XyxVik',
            sasl_plain_password=sasl_plain_password,
            group_id='$GROUP_NAME',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8').replace("'", '"'))
        )

    def poll_transactions(self, max_records: int = 100) -> list[Transaction]:
        """
        Polls for new messages and returns a list of up to max_records Transaction objects.
        """
        records = self.consumer.poll(timeout_ms=1000, max_records=max_records)

        polled_transactions = []
        for message in records.values():
            for record in message:
                print(record.value)
                transaction_obj = Transaction.from_object(record.value)
                polled_transactions.append(transaction_obj)

        return polled_transactions

    def close(self):
        self.consumer.close()


if __name__ == '__main__':
    consumer = KafkaTransactionConsumer('credit_card_transactions')
    transactions = consumer.poll_transactions()
    print(transactions)
