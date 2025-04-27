# Cybersecurity-Capstone-Spring-2025
# Botnet Traffic Intrusion Detection System

This project implements an Intrusion Detection System (IDS) that utilizes various Machine Learning (ML) algorithms and Neural Networks to classify network traffic data from the CTU-13 dataset as either normal or indicative of botnet activity. The application provides a graphical user interface (GUI) built with PyQt6, allowing users to easily select an algorithm, upload a dataset, train the model, and predict on new data.

## Features

* **Algorithm Selection:** Choose from a range of ML algorithms including:
    * Gaussian Naive Bayes
    * Random Forest
    * K Nearest Neighbors
    * Support Vector Machine
    * Logistic Regression
    * Long Short-Term Memory (LSTM)
    * Recurrent Neural Network (RNN)
    * Autoencoder
* **Dataset Upload:** Easily upload CSV files containing network traffic data (specifically designed for the CTU-13 dataset format).
* **Model Training and Evaluation:** Train the selected algorithm on the uploaded dataset. The system evaluates the model and displays performance metrics and visualizations.
* **Results Display:** A dedicated window shows the console output of the algorithm, evaluation metrics (Accuracy, Precision, Recall, F1-Score, False Negatives, False Positives), and generated plots such as:
    * Confusion Matrix
    * Receiver Operating Characteristic (ROC) Curve
    * Learning Curves (Accuracy and Loss, for applicable neural network models)
* **Prediction on New Data:** After training a model, users can upload a new, unseen dataset in the same format to predict whether the traffic is normal or malicious.
* **Prediction Results:** A summary of the predictions on the new dataset is displayed, showing the count of predicted normal and malicious traffic samples.
* **User-Friendly GUI:** An intuitive graphical interface built with PyQt6 makes the application easy to use for both technical and non-technical users.

## Project Structure

The project contains the following files:

* **`Datasets/`**: This folder contains the datasets used for training and testing the models.
* **`ML Algorithms/`**: This folder contains individual Python files for each of the implemented Machine Learning algorithms.
* **`Neural Network Algorithms/`**: This folder contains individual Python files for each of the implemented Neural Network models.
* `Main.py`: The main script containing the GUI implementation (`MainWindow`, `ResultsWindow`, `PredictionWindow`).
* `algorithms.py`: This file (`algorithms.py`) contains the implementations of the different machine learning algorithms and neural network models used for botnet detection. Each algorithm class inherits from a base `Model_` class which handles preprocessing, data splitting, and evaluation. The file includes classes for:
    * `Preprocessing`: Base class for data preprocessing.
    * `Model_`: Base class for machine learning models, inheriting from `Preprocessing`.
    * `GaussianNaiveBayes`
    * `RandomForest`
    * `KNearestNeighbors`
    * `SupportVectorMachine`
    * `LogisticRegression`
    * `LSTM`
    * `RNN`
    * `Autoencoder`

## Prerequisites

Before running the application, ensure you have Python 3 installed on your system. You will also need to install the required Python libraries.

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/EJ-Conner/Cybersecurity-Capstone-Spring-2025
    cd Cybersecurity-Capstone-Spring-2025
    ```

2.  **Install the required Python packages** using pip:
    ```bash
    pip install PyQt6 pandas numpy scikit-learn matplotlib tensorflow seaborn
    ```
    **Explanation of the pip installs:**
    * `PyQt6`: For creating the graphical user interface.
    * `pandas`: For data manipulation and working with CSV files.
    * `numpy`: For numerical computations and array operations.
    * `scikit-learn`: For various machine learning algorithms, model evaluation tools, and data preprocessing techniques.
    * `matplotlib`: For generating plots and visualizations (ROC curve, learning curves).
    * `tensorflow`: For implementing and running the neural network models (LSTM, RNN, Autoencoder).
    * `seaborn`: For enhanced and visually appealing statistical plots, used here for the confusion matrix.

## Usage

1.  **Run the main application script:**
    ```bash
    python Main.py
    ```

2.  **Follow the steps in the GUI:**
    * **Step 1: Select Algorithm:** Choose the desired machine learning algorithm or neural network from the dropdown menu.
    * **Step 2: Upload Dataset:** Click the "Click to Upload CSV Dataset" button and select your CTU-13 dataset file in CSV format. The status label will update to show the selected file.
    * **Step 3: Run Analysis:** Once a dataset is uploaded, the "Run Selected Algorithm" button will be enabled. Click it to start the training and evaluation process. A "Results" window will appear upon completion, displaying the algorithm's output, evaluation metrics, and generated plots.
    * **Step 4: Upload New Data for Prediction:** After a model has been successfully trained, the "Click to Upload Prediction CSV" button will be enabled. Click it to select a new CSV file (in the same format as the training data) for prediction.
    * **Step 5: Predict on New Data:** Once a prediction dataset is uploaded, the "Predict Data" button will become enabled. Click it to run the prediction using the trained model. A "Prediction Results" window will display a summary of the predicted classes (Normal or Malicious).

## Dataset

This project is designed to work with the **CTU-13 dataset**, a well-known dataset for botnet traffic analysis. The dataset files are located in the `Datasets/` directory. However, the application could be used with other datasets in a compatible CSV format.

## Acknowledgements

* The creators of the CTU-13 dataset for providing valuable data for botnet research.
* The developers of PyQt, pandas, numpy, scikit-learn, matplotlib, seaborn, and TensorFlow for their excellent libraries.
