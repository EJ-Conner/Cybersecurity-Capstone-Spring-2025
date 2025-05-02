import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the datasets
df1 = pd.read_csv("CTU13_Normal_Traffic.csv")
df2 = pd.read_csv("CTU13_Attack_Traffic.csv")
combined_df = pd.concat([df1, df2], ignore_index=True)

# Splitting into features and target variable
x = combined_df.iloc[:, :-1]
y = combined_df.iloc[:, -1]

x_scaled = StandardScaler().fit_transform(x) #Scale x

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=3)

# Initializing and training the KNN model
lg = LogisticRegression(max_iter = 250) #Increase max iterations to find parameters because of large dataset (100 is the default)
lg.fit(x_train, y_train)

# Predicting on the test set
y_pred = lg.predict(x_test)

# Calculating Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, lg.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Printing metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualizations
# Confusion Matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

