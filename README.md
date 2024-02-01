# Text-Summarizer Project

## Overview

This project is designed for summarizing conversations, akin to the Samsum Dataset. It is deployment-ready, featuring a user interface implemented with FastAPI. The key highlight is the fine-tuning of the `google/pegasus_cnn_daily` model on the Samsum dataset, which serves as the final model for text summarization.

## Project Structure

### 1. Training Pipeline

- **Data Ingestion:** Ingests data from the specified source, ensuring seamless integration with the summarization model.
  
- **Data Validation:** Performs validation on the acquired data to ensure its quality and adherence to expected formats.

- **Data Transformation:** Transforms the validated data into a suitable format for training the model.

- **Model Trainer:** Utilizes the transformed data to fine-tune the `google/pegasus_cnn_daily` model on the Samsum dataset.

- **Model Evaluation:** Assesses the performance of the trained model, providing insights into its effectiveness.

### 2. Prediction Pipeline

- **Text Input:** Takes text input for which summarization is required.

- **Inference:** Uses the finalized model to generate concise summaries based on the provided text input.

## Deployment

The entire project is containerized using docker and can be effortlessly deployed. Continuous integration and deployment are streamlined for a smooth development process.

## Usage

To use the project, follow these steps:

1. Clone the repository:

``git clone https://github.com/AbhayaHanuma/Text-Summerize.git``

2. Navigate to the project directory and run `docker build -t text-summarizer .` to build the Docker image
3. Run `docker run -p 8080:8080 text-summarizer` to start the FastAPI app
4. Open your browser and go to `http://localhost:8080` to see the app

## Contribution

Feel free to contribute by opening issues, suggesting improvements, or submitting pull requests.