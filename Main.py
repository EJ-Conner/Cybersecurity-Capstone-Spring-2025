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
        self.x = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
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
class Model(Preprocessing):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        #Preprocessing:
        self.x, self.y = self.preprocess_data()
        self.x = self.scale_features()
        if self.__class__.__name__ == 'LSTM':
            #Reshape data for LSTM [samples, time steps, features]
            self.x = np.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1]))
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data()

    def train(self):
        pass

    def evaluate(self):
        #Predicting on the test set
        y_pred = self.predict(self.x_test)

        #Calculating Accuracy, Precision, Recall, F1-Score
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average = 'binary')
        recall = recall_score(self.y_test, y_pred, average = 'binary')
        f1 = f1_score(self.y_test, y_pred, average = 'binary')

        #Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        false_positives = cm[0,1]
        false_negatives = cm[1,0]
        self.plot_confusion_matrix(cm)

        if self.__class__.__name__ != 'LSTM': #Use in ML algorithms
            #ROC Curve
            fpr, tpr, thresholds = roc_curve(self.y_test, self.predict_proba(self.x_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)
        
        #Printing metrics
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Flase negatives: {false_negatives}')
        print(f'False Positives: {false_positives}')

    #Confusion Matrix heatmap
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize = (6, 5))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Normal', 'Attack'], yticklabels = ['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

     #ROC Curve
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize = (6, 5))
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc = 'lower right')
        plt.show()

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

#Gaussian Naive Bayes class
class GaussianNaiveBayesModel(Model):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.naive_bayes import GaussianNB

        self.nb = GaussianNB()

    def train(self):
        self.nb.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.nb.predict(X)

    def predict_proba(self, X):
        return self.nb.predict_proba(X)

#Random Forrest class
class RandomForest(Model):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier()

    def train(self):
        self.rf.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.rf.predict(X)

    def predict_proba(self, X):
        return self.rf.predict_proba(X)
        
#KNN class
class KNN(Model):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.neighbors import KNeighborsClassifier

        self.knn = KNeighborsClassifier()

    def train(self):
        self.knn.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.knn.predict(X)

    def predict_proba(self, X):
        return self.knn.predict_proba(X)

#SVM class
class SVM(Model):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn import svm

        self.svm = svm.SVC(kernel = 'rbf', probability=True)

    def train(self):
        self.svm.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.svm.predict(X)

    def predict_proba(self, X):
        return self.svm.predict_proba(X)

#LSTM class
class LSTM(Model):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        #Model
        self.model = Sequential()
        #LSTM layers
        self.model.add(LSTM(units = 50, return_sequences = False, input_shape = (self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 50, activation = 'relu'))
        self.model.add(Dense(units = 1, activation = 'sigmoid'))  #'softmax' for multi-class classification
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    def train(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=64, validation_data=(self.x_test, self.y_test), verbose=2)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype(int) #Convert probabilities to binary class labels

class main():
    #algorithm = GaussianNaiveBayesModel('CTU13_Combined_Traffic.csv')
    algorithm = RandomForest('CTU13_Combined_Traffic.csv')
    #algorithm = KNN('CTU13_Combined_Traffic.csv')
    #algorithm = SVM('CTU13_Combined_Traffic.csv')
    #algorithm = LSTM('CTU13_Combined_Traffic.csv')
    algorithm.train()
    algorithm.evaluate()

if __name__ == '__main__':
    main()
