import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

#Base class for machine learning models
class Model:
    def __init__(self, dataset_path1, dataset_path2):
        #Load the datasets
        self.df1 = pd.read_csv(dataset_path1)
        self.df2 = pd.read_csv(dataset_path2)
        self.combined_df = pd.concat([self.df1, self.df2], ignore_index=True)

        #Split into features and target variable
        self.x = self.combined_df.iloc[:, :-1]
        self.y = self.combined_df.iloc[:, -1]

        #Split the dataset into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=3)

    def train(self):
        pass

    def evaluate(self):
        #Predicting on the test set
        y_pred = self.predict(self.x_test)

        #Calculating Accuracy, Precision, Recall, F1-Score
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')

        #Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)

        #ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.predict_proba(self.x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        #Printing metrics
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        #Graphs
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(fpr, tpr, roc_auc)

    #Confusion Matrix heatmap
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

     #ROC Curve
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

#Gaussian Naive Bayes class
class GaussianNaiveBayesModel(Model):
    def __init__(self, dataset_path1, dataset_path2):
        super().__init__(dataset_path1, dataset_path2)
        from sklearn.naive_bayes import GaussianNB
        self.nb = GaussianNB()

        #Scale data for Gaussian Naive Bayes
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self):
        self.nb.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.nb.predict(X)

    def predict_proba(self, X):
        return self.nb.predict_proba(X)

#Random Forrest class
class RandomForrest(Model):
    def __init__(self, dataset_path1, dataset_path2):
        super().__init__(dataset_path1, dataset_path2)
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
    def __init__(self, dataset_path1, dataset_path2):
        super().__init__(dataset_path1, dataset_path2)
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
    def __init__(self, dataset_path1, dataset_path2):
        super().__init__(dataset_path1, dataset_path2)
        from sklearn import svm
        self.svm = svm.SVC(kernel='rbf', probability=True)

    def train(self):
        self.svm.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.svm.predict(X)

    def predict_proba(self, X):
        return self.svm.predict_proba(X)

class main():
    #algorithm = GaussianNaiveBayesModel("CTU13_Normal_Traffic.csv", "CTU13_Attack_Traffic.csv")
    #algorithm = RandomForrest('CTU13_Normal_Traffic.csv', 'CTU13_Attack_Traffic.csv')
    #algorithm = KNN('CTU13_Normal_Traffic.csv', 'CTU13_Attack_Traffic.csv')
    algorithm = SVM('CTU13_Normal_Traffic.csv', 'CTU13_Attack_Traffic.csv')
    algorithm.train()
    algorithm.evaluate()

if __name__ == '__main__':
    main()
