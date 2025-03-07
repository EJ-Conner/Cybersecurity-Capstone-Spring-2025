import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

# Set random seed for reproducibility
RANDOM_SEED = 42
LABELS = ["Normal", "Attack"]

# Load Botnet Detection Dataset
df1 = pd.read_csv("CTU13_Normal_Traffic.csv")  # Normal traffic
df2 = pd.read_csv("CTU13_Attack_Traffic.csv")  # Attack traffic

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Ensure the last column is the label (0 = Normal, 1 = Attack)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Convert labels if they are not numeric
if y.dtype == 'object':
    y = pd.factorize(y)[0]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

# Train autoencoder **only on normal traffic** (y_train == 0)
X_train_normal = X_train[y_train == 0]

# Autoencoder Architecture
input_dim = X_train_normal.shape[1]
encoding_dim = 9  # Compressed representation

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile Model
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Model Training
nb_epoch = 20
batch_size = 16

checkpointer = ModelCheckpoint(filepath="botnet_model.h5", save_best_only=True, verbose=0)
tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=nb_epoch,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1,
    callbacks=[checkpointer, tensorboard]
)

# Load best model
autoencoder = load_model('botnet_model.h5')

# Plot Training Loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Autoencoder Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# **Reconstruction Error Analysis**
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

# **Histogram of Reconstruction Error**
plt.hist(error_df[error_df.true_class == 0]['reconstruction_error'], bins=50, alpha=0.5, label="Normal")
plt.hist(error_df[error_df.true_class == 1]['reconstruction_error'], bins=50, alpha=0.5, label="Attack")
plt.yscale('log')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency (Log Scale)")
plt.legend()
plt.show()

# **ROC Curve Analysis**
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# **Precision-Recall Curve**
precision, recall, _ = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# **Threshold Selection**
threshold = np.percentile(error_df[error_df.true_class == 0]['reconstruction_error'], 95)  # 95th percentile of normal

# **Classify Botnet vs. Normal**
error_df["predicted"] = (error_df.reconstruction_error > threshold).astype(int)

# **Confusion Matrix**
conf_matrix = confusion_matrix(error_df.true_class, error_df.predicted)
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# **Final Classification Report**
print("Classification Report:\n", classification_report(error_df.true_class, error_df.predicted))
