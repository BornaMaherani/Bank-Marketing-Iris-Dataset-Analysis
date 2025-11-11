# Machine Learning Homework 1

This repository contains solutions and implementations for Machine Learning Homework 1. The project focuses on two main tasks:

1.  **Bank Marketing Dataset Analysis (Q1.ipynb):** This notebook analyzes a Bank Marketing dataset to predict whether a client will subscribe to a term deposit. It involves data preprocessing using `sklearn.pipeline`, training and evaluating Decision Tree and Random Forest classifiers, and hyperparameter tuning.

2.  **K-Nearest Neighbors (KNN) Algorithm Implementation (knn.py & Q2.ipynb):** This section provides a custom implementation of the K-Nearest Neighbors (KNN) algorithm using NumPy (`knn.py`). The `Q2.ipynb` notebook then tests, optimizes, and evaluates this custom KNN model on the Iris dataset, including hyperparameter tuning with `GridSearchCV` and cross-validation.

## Project Structure

-   `bank.csv`: The dataset used for the Bank Marketing analysis.
-   `knn.py`: Contains the `MyKNNClassifier` class, a custom implementation of the K-Nearest Neighbors algorithm.
-   `Q1.ipynb`: Jupyter notebook for the Bank Marketing dataset analysis.
-   `Q2.ipynb`: Jupyter notebook for testing and evaluating the custom KNN implementation on the Iris dataset.
-   `README.md`: This file.

## Setup and Usage

To run the notebooks and code in this repository, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BornaMaherani/Bank-Marketing-Iris-Dataset-Analysis.git
    cd ML-HW1
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided, but typically would list `numpy`, `pandas`, `scikit-learn`, `jupyter`.)*

3.  **Run Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```
    This will open a browser window where you can navigate to `Q1.ipynb` and `Q2.ipynb` to view and run the analyses.

## `MyKNNClassifier` Details

The `knn.py` file implements a `MyKNNClassifier` class with the following features:

-   **`__init__(self, n_neighbors=3, metric="euclidean", weighted=False)`**:
    -   `n_neighbors`: Number of neighbors to consider (default: 3).
    -   `metric`: Distance metric to use ("euclidean" or "manhattan", default: "euclidean").
    -   `weighted`: If True, uses weighted voting based on inverse distance (default: False).
-   **`fit(self, X, y)`**: Stores the training data.
-   **`_calculate_distances(self, x)`**: Internal method to calculate distances from a sample `x` to all training data points using the specified metric.
-   **`_predict_one(self, x)`**: Internal method to predict the label for a single sample using k-nearest neighbors and voting (weighted or unweighted).
-   **`predict(self, X)`**: Predicts labels for an array of samples `X`.

## Analysis Highlights

-   **Q1.ipynb**: Demonstrates a typical machine learning workflow including data loading, preprocessing (scaling numerical features, one-hot encoding categorical features), model training (Decision Tree, Random Forest), and evaluation using accuracy, precision, and recall. It also includes a section on hyperparameter tuning for the Decision Tree.
-   **Q2.ipynb**: Focuses on the custom KNN implementation. It loads and preprocesses the Iris dataset, then uses `GridSearchCV` with `KFold` cross-validation to find optimal hyperparameters for `MyKNNClassifier` (k, distance metric, and weighting). The final model performance is reported using a classification report.
