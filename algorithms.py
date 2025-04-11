import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


#Base Preprocessing Class
class Preprocessing:
    def __init__(self, dataset_path):
        #Load the dataset
        self.df = pd.read_csv(dataset_path)
        
    def preprocess_data(self):
        #Handle missing values by filling with mean
        self.df.fillna(self.df.mean(), inplace = True)
        
        #Encode categorical target column (assumes the last column is the target)
        self.label_encoder = LabelEncoder()
        self.df.iloc[:, -1] = self.label_encoder.fit_transform(self.df.iloc[:, -1])

        #Split into features and target variable
        self.x = self.df.iloc[::100, :-1]
        self.y = self.df.iloc[::100, -1]
        return self.x, self.y

    def scale_features(self):
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)
        return self.x

    def split_data(self, test_size = 0.25):
        #Split the dataset into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size)
        return self.x_train, self.x_test, self.y_train, self.y_test

#Base class for machine learning models
class Model_(Preprocessing):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        print(f"Model_ __init__ called for {self.__class__.__name__}") # Add this line

        #Preprocessing:
        self.x, self.y = self.preprocess_data()
        self.x = self.scale_features()
        if self.__class__.__name__ == 'LSTM' or self.__class__.__name__ == 'RNN':
            #Reshape data for LSTM [samples, time steps, features]
            self.x = np.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1]))
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data()
        self.confusion_chart = None
        self.roc_chart = None

    def train(self):
        pass

        #method to get the algorithm-specific output directory
    def get_output_dir(self):
        base_output_dir = "all_outputs"
        #Generate a folder name from the class name
        algo_name = self.__class__.__name__
        specific_dir = os.path.join(base_output_dir, f"{algo_name.replace(' ', '')}_output")
        os.makedirs(specific_dir, exist_ok=True) # Create dir if it doesn't exist
        return specific_dir

    def evaluate(self):

        # Get the specific output directory for algorithm
        output_dir = self.get_output_dir()

        #Predicting on the test set
        y_pred, y_pred_prob = self.predict(self.x_test)

        #Calculating Accuracy, Precision, Recall, F1-Score
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average = 'binary')
        recall = recall_score(self.y_test, y_pred, average = 'binary')
        f1 = f1_score(self.y_test, y_pred, average = 'binary')



        #Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        if cm.shape == (2, 2):
            false_positives = cm[0, 1]
            false_negatives = cm[1, 0]
        else: # Handle case where predictions are all one class, get counts safely
            try:
                tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred, labels=[0, 1]).ravel()
                false_positives = fp
                false_negatives = fn
            except ValueError: # If only one class exists in y_test *and* y_pred
                print("Warning: Could not calculate FN/FP due to uniform predictions/labels.")
                false_positives = "N/A"
                false_negatives = "N/A"

        # saving conf matrix 
        try:
             plt.figure(figsize=(6, 5))
             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
             plt.ylabel('Actual')
             plt.xlabel('Predicted')
             plt.title('Confusion Matrix')
             cm_path = os.path.join(output_dir, "confusion_matrix.png")
             plt.savefig(cm_path)
             plt.close() # Close the figure
             print(f"Saved confusion matrix to: {cm_path}") # Optional debug print
        except Exception as e:
             print(f"Error saving confusion matrix: {e}")

        # --- Save ROC Curve using Matplotlib ---
        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            roc_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close() # Close the figure
            print(f"Saved ROC curve to: {roc_path}") # Optional debug print
        except Exception as e:
            print(f"Error saving ROC curve: {e}")
                
        
        metrics_contents = (f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nFalse negatives: {false_negatives}\nFalse Positives: {false_positives}')
        try:
            metrics_path = os.path.join(output_dir, "metrics.txt")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as file:
                file.write(metrics_contents)
                print(f'File was successfully saved to {metrics_path}')
        except Exception as e:
            print(f"Error saving metrics.txt: {e}")
        

        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_negatives': false_negatives,
            'false_positives': false_positives
        }
        

    def _save_learning_curves(self, history):
        # Saves accuracy and loss learning curves using Matplotlib to the algorithm's specific folder.
        output_dir = self.get_output_dir() # Get the correct directory

        # Plot and save Accuracy curve
        if hasattr(history, 'history') and 'accuracy' in history.history and 'val_accuracy' in history.history:
            try:
                plt.figure()
                plt.plot(history.history['accuracy'], label='Train Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend()
                acc_path = os.path.join(output_dir, "learning_curve_accuracy.png")
                plt.savefig(acc_path)
                plt.close()
                print(f"Saved accuracy curve to: {acc_path}") # Optional debug print
            except Exception as e:
                print(f"Error saving accuracy curve: {e}")


        # Plot and save Loss curve
        if hasattr(history, 'history') and 'loss' in history.history and 'val_loss' in history.history:
            try:
                plt.figure()
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend()
                loss_path = os.path.join(output_dir, "learning_curve_loss.png")
                plt.savefig(loss_path)
                plt.close()
                print(f"Saved loss curve to: {loss_path}") # Optional debug print
            except Exception as e:
                print(f"Error saving loss curve: {e}")
                
    def predict(self, X):
        pass

#Gaussian Naive Bayes class
class GaussianNaiveBayes(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.naive_bayes import GaussianNB

        self.nb = GaussianNB()

    def train(self):
        self.nb.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.nb.predict(X), self.nb.predict_proba(X)[:, 1]

#Random Forrest class
class RandomForest(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier()

    def train(self):
        self.rf.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.rf.predict(X), self.rf.predict_proba(X)[:, 1]
        
#KNN class
class KNearestNeighbors(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.neighbors import KNeighborsClassifier

        self.knn = KNeighborsClassifier()

    def train(self):
        self.knn.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.knn.predict(X), self.knn.predict_proba(X)[:, 1]

#SVM class
class SupportVectorMachine(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn import svm

        self.svm = svm.SVC(kernel = 'rbf', probability=True)

    def train(self):
        self.svm.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.svm.predict(X), self.svm.predict_proba(X)[:, 1]

#Logistic Regression class
class LogisticRegression(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.linear_model import LogisticRegression

        self.lg = LogisticRegression(max_iter = 250) #Increase max iterations to find parameters because of large dataset (100 is the default)

    def train(self):
        self.lg.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.lg.predict(X), self.lg.predict_proba(X)[:, 1]


#LSTM class
class LSTM(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        # --- For different types of envirnoments
        # Microsoft Visual Studio
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.layers import LSTM, Dense, Dropout

        # Conda Environment
        import tensorflow as tf 
        from tensorflow import keras

        #Model
        self.model = keras.models.Sequential()
        #LSTM layers
        self.model.add(keras.layers.LSTM(units = 50, return_sequences = False, input_shape = (self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units = 50, activation = 'relu'))
        self.model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))  #'softmax' for multi-class classification
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    def train(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=64, validation_data=(self.x_test, self.y_test), verbose=2)
        self._save_learning_curves(self.history)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype(int), y_pred #Convert probabilities to binary class labels

#RNN class
class RNN(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        # --- For different types of envirnoments
        # Microsoft Visual Studio
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
        
        # Conda Environment
        import tensorflow as tf 
        from tensorflow import keras

        #Model
        self.model = keras.models.Sequential()
        #RNN layers
        self.model.add(keras.layers.SimpleRNN(units=50, return_sequences=False, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units=50, activation='relu'))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  #Use 'categorical_crossentropy' for multi-class

    def train(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=64, validation_data=(self.x_test, self.y_test), verbose=2)
        self._save_learning_curves(self.history)

    def predict(self, X):
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class labels
        return y_pred_classes, y_pred

#Autoencodder class
class Autoencoder(Model_):
  
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        # --- For different types of envirnoments
        # Microsoft Visual Studio
        # from tensorflow.keras.models import Model
        # from tensorflow.keras.layers import Input, Dense
        
        # Conda Environment
        import tensorflow as tf 
        from tensorflow import keras
        from tensorflow.keras.regularizers import l1

        # Train autoencoder **only on normal traffic** (y_train == 0)
        self.X_train_normal = self.x_train[self.y_train == 0]

        # Autoencoder Architecture
        input_dim = self.X_train_normal.shape[1]
        encoding_dim = 9  # Compressed representation
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoder = keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=l1(10e-5))(input_layer)
        encoder = keras.layers.Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = keras.layers.Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = keras.layers.Dense(input_dim, activation='relu')(decoder)
        self.autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self):

        # --- For different types of envirnoments
        # Microsoft Visual Studio
        #from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
        
        # Conda Environment
        from tensorflow import keras

        output_dir = self.get_output_dir() #Get specific dir for checkpoint path
        logs_dir = os.path.join(output_dir, 'logs') #Place logs inside algorithm output dir
        os.makedirs(logs_dir, exist_ok=True)

        nb_epoch = 20
        batch_size = 16
        checkpointer = keras.callbacks.ModelCheckpoint(filepath = "botnet_model.h5", save_best_only = True, verbose = 0)
        tensorboard = keras.callbacks.TensorBoard(log_dir = './logs', write_graph = True, write_images = True)

        self.history = self.autoencoder.fit(
            self.X_train_normal, self.X_train_normal,
            epochs = nb_epoch,
            batch_size = batch_size,
            shuffle = True,
            validation_data = (self.x_test, self.x_test),
            verbose = 1,
            callbacks = [checkpointer, tensorboard]
        )
        self._save_learning_curves(self.history)

    def predict(self, X):
        predictions = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred = (mse > threshold).astype(int)
        y_pred_prob = mse / np.max(mse)
        return y_pred, y_pred_prob
