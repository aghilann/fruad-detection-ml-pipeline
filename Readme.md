# ML Infra - Credit Card Fraud Detection

<!-- ![Project Logo](images/logo.png) -->

## Introduction

The Credit Card Fraud Detection project is a machine learning-based solution that aims to identify fraudulent credit card transactions from a dataset of credit card transactions. It utilizes Apache Spark and a Random Forest classification model to predict whether a transaction is fraudulent or not. It uses Flyte for orchestrating the machine learning workflow and Flask for serving the model as a REST API.

## Features

- Efficiently detects fraudulent credit card transactions.
- Utilizes Apache Spark for distributed data processing.
- Easy-to-use REST API for making real-time predictions.
- Supports batch prediction for multiple transactions.

## Technologies

- Python
- Apache Spark
- Flyte
- PySpark
- Flask
- Pandas
- Scikit-learn

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher installed
- Java 8 or higher installed (for Apache Spark)

### Model Stats

```
Model Accuracy: 0.9958040104826124                                              
Model Precision: 0.9956810532257646
Model Recall: 0.9958040104826125
Model F1-Score: 0.9945218711604894
```

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   ```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python workflow.py
   ```