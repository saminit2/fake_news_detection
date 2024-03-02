# News Article Classifier

This repository contains code for classifying news articles as either real or fake using machine learning models. The dataset used for training and testing the models can be downloaded from [this link](https://drive.google.com/file/d/1ygYA7uHAOt4Ix_hKT6Xdcuj94SCrm3Tn/view?usp=sharing).

## About the Dataset
The dataset consists of news articles along with their corresponding labels indicating whether the news is real or fake.

## Code Overview
The code is organized into several sections to facilitate understanding and usage. Here's a brief overview:

1. **Data Loading and Preparation**: The dataset is loaded from the provided file path and split into features (X) and labels (Y). Train-test split is performed to create training and testing sets.

2. **Feature Engineering**: Text data is vectorized using the TF-IDF Vectorizer, which converts the text into numerical features.

3. **Model Training**: A Passive Aggressive Classifier is trained on the TF-IDF transformed data to classify news articles.

4. **Model Evaluation**: The trained model is evaluated using the testing set, and accuracy is calculated. Additionally, a confusion matrix is generated to visualize the performance of the classifier.

5. **Model Persistence**: The trained model is saved using the pickle module for future use.

6. **Prediction**: The user can input news articles interactively to get predictions from the trained model.

## Instructions
1. **Clone the Repository**: Clone this repository to your local machine using `git clone`.
2. **Install Dependencies**: Install the required dependencies listed in `requirements.txt` using `pip install -r requirements.txt`.
3. **Download the Dataset**: Download the dataset file from the provided [Google Drive link](https://drive.google.com/file/d/1ygYA7uHAOt4Ix_hKT6Xdcuj94SCrm3Tn/view?usp=sharing).
4. **Run the Code**: Execute the `news.py` script to train the model, evaluate its performance, and make predictions.

## File Structure
- `news.py`: Main Python script containing the code for data loading, preprocessing, model training, evaluation, and prediction.
- `README.md`: This file containing information about the project.

## Tech Stack
The project is implemented using:
- Python
- pandas
- scikit-learn
- matplotlib

## Contributors
- [Your Name](https://github.com/yourusername)

Feel free to contribute to this project by submitting issues or pull requests!


