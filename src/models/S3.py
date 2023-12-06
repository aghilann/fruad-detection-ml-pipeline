import os
from typing import Union

import boto3


class S3:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.client('s3')

    def read_file(self, folder_path: str, file_name: str) -> str:
        file_key = f"{folder_path}/{file_name}"
        obj = self.client.get_object(Bucket=self.bucket_name, Key=file_key)
        return obj['Body'].read().decode('utf-8')

    def read_binary(self, folder_path: str, file_name: str) -> bytes:
        file_key = f"{folder_path}/{file_name}"
        assert file_key == "data/credit_card_fraud_model.tar.gz"
        assert self.bucket_name == "mlinfra"
        print("GET X", file_key)
        obj = self.client.get_object(Bucket=self.bucket_name, Key=file_key)
        return obj['Body'].read()

    def write_file(self, folder_path: str, file_name: str, content: Union[str, bytes]):
        file_key = f"{folder_path}/{file_name}"
        if isinstance(content, str):
            content = content.encode('utf-8')
        self.client.put_object(Bucket=self.bucket_name, Key=file_key, Body=content)

    def download_file(self, folder: str, file_name: str, local_path: str = '/tmp/') -> str:
        """
        Download a file from S3 to a local directory.

        :param folder: The folder within the S3 bucket where the file is located.
        :param file_name: The name of the file on S3.
        :param local_path: The local directory to which the file should be downloaded.
        :return: The path to the locally saved file.
        """
        local_file_path = os.path.join(local_path, file_name)

        # Download file
        self.client.download_file(self.bucket_name, f"{folder}/{file_name}", local_file_path)

        return local_file_path


if __name__ == '__main__':
    pass
    s3_store = S3(bucket_name='mlinfra')
    x = s3_store.read_file('data', 'credit_card_fraud.csv')

