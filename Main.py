

import os
import sys
import traceback
import io
import contextlib
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QScrollArea, QWidget, QVBoxLayout, QComboBox, 
                            QPushButton, QApplication, QFileDialog, QLabel,
                            QTextEdit, QDialog, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

try:
    import algorithms
except ImportError:
    print("Error: algorithms.py not found. Make sure it's in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error importing algorithms.py: {e}")
    traceback.print_exc() # Print detailed traceback
    sys.exit(1)


def add_plot_to_layout(plot_path, title, layout_to_add_to, scale_width=600, scale_height=450):
    # ---- Adds a plot image to the specified layout if it exists ----
    plot_widget = QWidget() #Container for title + image
    plot_layout = QVBoxLayout(plot_widget)
    plot_layout.setContentsMargins(0, 5, 0, 5) # margins

    plot_title_label = QLabel(f"<b>{title}</b>")
    plot_layout.addWidget(plot_title_label)

    if plot_path and os.path.exists(plot_path):
        plot_label = QLabel()
        pixmap = QPixmap(plot_path)
        #Scale pixmap smoothly
        scaled_pixmap = pixmap.scaled(scale_width, scale_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        plot_label.setPixmap(scaled_pixmap)
        plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(plot_label)
        print(f"Successfully added plot '{title}' from {plot_path}")
    else:
        print(f"Error: Plot not found or path not provided for '{title}': {plot_path}")
        missing_label = QLabel(f"<i>'{title}' plot not generated or found at expected path.</i>")
        missing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(missing_label)

    layout_to_add_to.addWidget(plot_widget) # Add the container widget


# ------- Displays Results Window After Running Algorithm -------
class ResultsWindow(QDialog):
    def __init__(self, output_text, parent=None, metrics_text=None,
                 selected_algorithm=None, algorithm_output_dir=None, is_error=False):
        super().__init__(parent)
        self.setWindowTitle(f"{selected_algorithm} Results" if selected_algorithm else "Results")
        self.setMinimumSize(850, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        #Store passed info from main window
        self.selected_algorithm = selected_algorithm
        self.algorithm_output_dir = algorithm_output_dir
        self.is_error = is_error 

        # --- Scrollable Window ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #ffffff; }") #White background
        scroll_area_widget = QWidget()
        scroll_area_widget.setStyleSheet("background-color: #ffffff;")
        self.scroll_layout = QVBoxLayout(scroll_area_widget)  #Layout for scrollable content
        self.scroll_layout.setSpacing(20)                     #Create spacing between sections
        self.scroll_layout.setContentsMargins(10, 10, 10, 10) #Margins inside scroll area
        scroll_area.setWidget(scroll_area_widget)
        layout.addWidget(scroll_area, 1)                      # Add scroll area to the main dialog layout


        # --- Algorithm Console Output --- (either error if error flag is true or correct output)
        console_label_text = "<b>Error Details:</b>" if self.is_error else "<b>Algorithm Console Output:</b>"
        results_label = QLabel(console_label_text)
        self.scroll_layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText(output_text)
        #Flexible height
        self.results_text.setMinimumHeight(150)
        #self.results_text.setMaximumHeight(300) #Limit max initial height
        self.results_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.scroll_layout.addWidget(self.results_text)

        

        # --- Metrics Display (Only if not an error window and metrics exist) ---
        if not self.is_error and metrics_text:
            metrics_label_title = QLabel("<b>Evaluation Metrics:</b>")
            self.scroll_layout.addWidget(metrics_label_title)
            metrics_display = QTextEdit()
            metrics_display.setReadOnly(True)
            metrics_display.setText(metrics_text)
            metrics_display.setFixedHeight(120) #Fixed height for metrics
            metrics_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.scroll_layout.addWidget(metrics_display)
        elif not self.is_error:
             #Still add a label if metrics are missing in a normal run, but notify user
             missing_metrics = QLabel("<i>Metrics file (metrics.txt) not found or not generated.</i>")
             missing_metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
             self.scroll_layout.addWidget(missing_metrics)


        # --- Plots Section (Only if not an error window and output dir exists) ---
        if not self.is_error and self.algorithm_output_dir:
            plots_section_label = QLabel("<b>Generated Plots:</b>")
            plots_section_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plots_section_label.setStyleSheet("font-size: 16px; margin-top: 10px; margin-bottom: 5px;")
            self.scroll_layout.addWidget(plots_section_label)

            print(f"Results window checking for plots in: {self.algorithm_output_dir}\n")
            # Define paths to saved image outputs
            confusion_matrix_path = os.path.join(self.algorithm_output_dir, "confusion_matrix.png")
            roc_curve_path = os.path.join(self.algorithm_output_dir, "roc_curve.png")
            accuracy_curve_path = os.path.join(self.algorithm_output_dir, "learning_curve_accuracy.png")
            loss_curve_path = os.path.join(self.algorithm_output_dir, "learning_curve_loss.png")

            # Adding Confusion Matrix to the plot section 
            add_plot_to_layout(confusion_matrix_path, "Confusion Matrix", self.scroll_layout)

            # Adding ROC Curve to the plot section
            add_plot_to_layout(roc_curve_path, "ROC Curve", self.scroll_layout)

            # Conditionally attempt adding Learning Curves based on algorithm name
            learning_curve_algorithms = ["LSTM", "RNN", "Autoencoder"]
            if self.selected_algorithm in learning_curve_algorithms:
                print(f"Attempting to load learning curves for {self.selected_algorithm}")
                # Add Accuracy Curve to plot section
                add_plot_to_layout(accuracy_curve_path, "Learning Curve (Accuracy)", self.scroll_layout)
                # Add Loss Curve to plot section 
                add_plot_to_layout(loss_curve_path, "Learning Curve (Loss)", self.scroll_layout)
            else:
                 print(f"Learning curves not applicable/expected for {self.selected_algorithm}")

        self.scroll_layout.addStretch(1) # Pushes content up if it's short

        # --- Close Button ---
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(75) # Button Size
        close_button.setStyleSheet("background-color: red; color: white;") 
        close_button.clicked.connect(self.accept) 
        layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight) #Add to layout, aligned right



# ------- Displays Prediction Results -------
class PredictionWindow(QDialog):
    def __init__(self, predictions_class, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Results")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title_label = QLabel("<b>Prediction Summary:</b>")
        layout.addWidget(title_label)

        results_text = QTextEdit()
        results_text.setReadOnly(True)

        if predictions_class is None or len(predictions_class) == 0:
            summary = "No predictions were generated."
        else:
            # It assumes 0 is Normal, 1 is Malicious/Attack
            # Map predictions to labels
            labels = ["Normal" if p == 0 else "Malicious" for p in predictions_class.flatten()] # Flatten in case it's column vector
            normal_count = labels.count("Normal")
            malicious_count = labels.count("Malicious")
            total_count = len(labels)

            summary = f"Total Samples Processed: {total_count}\n\n"
            summary += f"Predicted Normal Traffic: {normal_count}\n"
            summary += f"Predicted Malicious Traffic: {malicious_count}\n\n"

        results_text.setText(summary)
        layout.addWidget(results_text)

        close_button = QPushButton("Close")
        close_button.setStyleSheet("background-color: red; color: white;")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight)




# ------- Displays Main Initial Window -------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.trained_model_instance = None  # To store trained model instance
        self.predict_dataset_path = None    

        # --- Window setup ---
        self.setWindowTitle("Botnet Traffic Analysis")
        self.setMinimumSize(500, 550) # Set minimum size
        #self.setMaximumSize(650, 700) # Set maximum size
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0; /* Light grey background */
            }
            QWidget {
                /* Using Segoe UI if available, otherwise use Arial */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px; 
                color: #333333; /* Dark grey text */
            }
            QLabel {
                margin-bottom: 4px; /* Space below labels */
                font-weight: bold; /* Make labels bold */
            }
             QLabel#DatasetStatusLabel, QLabel#PredictDatasetStatusLabel { /* For dataset label */
                font-weight: normal;
                margin-top: 5px;
                margin-bottom: 10px;
             }
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                min-height: 25px; /* Ensure decent height */
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            /* Can't figure out how to get arrow on drop down
            
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #555;
                margin-right: 6px;
                background-color: transparent; 
                border-bottom: none; 
                border-left-width: 4px; 
                border-right-width: 4px;
                border-top-width: 5px;
            }
            */
            
            QPushButton {
                border-radius: 5px;
                background-color: #4CAF50; /* Primary Green */
                color: white;
                padding: 9px 18px; /* Increased padding */
                font-weight: bold;
                border: none;
                min-height: 25px; /* Ensure decent height */
                outline: none; /* Remove focus outline */
            }
            QPushButton:hover {
                background-color: #45a049; /* Darker Green on hover */
            }
            QPushButton:pressed {
                background-color: #367c39; /* Even darker Green when pressed */
            }
            QPushButton:disabled {
                background-color: #cccccc; /* Grey when disabled */
                color: #666666;
            }
            QTextEdit { /* Style for text edits in ResultsWindow */
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff; /* White background */
                padding: 5px;
            }
            QDialog { /* Style for the ResultsWindow dialog */
                 background-color: #f8f8f8; /* Slightly off-white */
            }
        """)

        # --- Main widget and layout ---
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)           # Apply layout to container
        layout.setContentsMargins(25, 25, 25, 25) # Generous margins
        layout.setSpacing(15)                     # Spacing between widget groups

        # --- 1. Algorithm selection ---
        layout.addWidget(QLabel("1. Select Algorithm:"))
        self.algorithm_combo = QComboBox()
        # Algorithm Options
        self.algorithm_display_names = [
            "Gaussian Naive Bayes", "Random Forest", "K Nearest Neighbors",
            "Support Vector Machine", "Logistic Regression", "LSTM", "RNN", "Autoencoder"
        ]

        self.algorithm_class_names = [
            "GaussianNaiveBayes", "RandomForest", "KNearestNeighbors",
            "SupportVectorMachine", "LogisticRegression", "LSTM", "RNN", "Autoencoder"
        ]

        # Create the map using the actual class names for lookup, keyed by display name
        self.algorithm_map = {}
        found_classes = 0
        for display_name, class_name in zip(self.algorithm_display_names, self.algorithm_class_names):
            if hasattr(algorithms, class_name):
                self.algorithm_map[display_name] = getattr(algorithms, class_name)
                found_classes += 1
            else:
                # Add a warning if a class defined here isn't found in algorithms.py
                print(f"Warning: Class '{class_name}' not found in algorithms.py for display name '{display_name}'")

        # Check if the number of found classes matches expected
        if found_classes != len(self.algorithm_class_names):
             print(f"Warning: Expected {len(self.algorithm_class_names)} algorithm classes, but only found {found_classes} in algorithms.py.")
             print(f"Mapped display names: {list(self.algorithm_map.keys())}")


        # add algorithm names to combo box
        self.algorithm_combo.addItems(self.algorithm_display_names)

        layout.addWidget(self.algorithm_combo)

        # --- 2. Dataset Upload ---
        layout.addWidget(QLabel("2. Upload Dataset:"))
        self.upload_button = QPushButton("Click to Upload CSV Dataset")
        self.upload_button.clicked.connect(self.upload_dataset)
        layout.addWidget(self.upload_button)

        # Dataset status label
        self.dataset_label = QLabel("No dataset selected.")
        self.dataset_label.setObjectName("DatasetStatusLabel") 
        self.dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dataset_label.setStyleSheet("font-style: italic; color: #555;") #Gray
        layout.addWidget(self.dataset_label)


        # --- 3. Run Analysis ---
        layout.addWidget(QLabel("3. Run Analysis:"))
        self.run_button = QPushButton("Run Selected Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)  # Disabled until dataset is uploaded
        layout.addWidget(self.run_button)

        layout.addStretch(1) # Add flexible space to push run button down

         # --- 4. Upload Prediction Data --- 
        layout.addWidget(QLabel("4. Upload New Data for Prediction:"))
        self.upload_predict_button = QPushButton("Click to Upload Prediction CSV")
        self.upload_predict_button.clicked.connect(self.upload_predict_dataset)
        self.upload_predict_button.setEnabled(False) # Disabled until prediction dataset is uploaded
        layout.addWidget(self.upload_predict_button)

        # Prediction Dataset status label 
        self.predict_status_label = QLabel("No prediction dataset selected.")
        self.predict_status_label.setObjectName("PredictDatasetStatusLabel")
        self.predict_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.predict_status_label.setStyleSheet("font-style: italic; color: #555;") #Gray
        layout.addWidget(self.predict_status_label)

        # --- 5. Run Prediction --- 
        layout.addWidget(QLabel("5. Predict on New Data:"))
        self.predict_button = QPushButton("Predict Data")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False) # Disabled initially
        layout.addWidget(self.predict_button)



    def upload_dataset(self):
        # Use a more specific filter and allow All Files as fallback
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Dataset",
            "", # Start directory is default
            "CSV Files (*.csv);;All Files (*)" # Filter for CSV files
        )
        if file_name:
            # Check extension case-insensitively
            if file_name.lower().endswith('.csv'):
                self.dataset_path = file_name
                # Display only the filename
                base_name = os.path.basename(file_name)
                self.dataset_label.setText(f"Dataset: {base_name}")
                # Use style sheet to show uploaded dataset
                self.dataset_label.setStyleSheet("color: #2E8B57; font-weight: bold; font-style: normal;") # SeaGreen, bold
                self.run_button.setEnabled(True) # Enable run button
            else:
                self.dataset_path = None
                self.dataset_label.setText("Invalid file type. Please upload a CSV file.")
                self.dataset_label.setStyleSheet("color: #DC143C; font-weight: bold; font-style: normal;") # Crimson red, bold
                self.run_button.setEnabled(False)
                QMessageBox.warning(self, "Invalid File", "Please select a valid CSV file (.csv extension).")
        # else: keep previous state


    # --- Function to upload the dataset for prediction ---
    def upload_predict_dataset(self):
        # Similar to upload_dataset, but updates different variables/widgets
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Dataset for Prediction", 
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_name:
            if file_name.lower().endswith('.csv'):
                self.predict_dataset_path = file_name
                # Update prediction status label
                self.predict_status_label.setText(f"Predict Data: {os.path.basename(file_name)}")
                self.predict_status_label.setStyleSheet("font-style: normal; font-weight: bold; color: green;")
                # Enable the predict button ONLY if a model is also trained
                if self.trained_model_instance:
                    self.predict_button.setEnabled(True)
            else:
                self.predict_dataset_path = None
                self.predict_status_label.setText("Invalid file type. Please select a CSV.")
                self.predict_status_label.setStyleSheet("font-style: italic; font-weight: bold; color: red;")
                self.predict_button.setEnabled(False)
                QMessageBox.warning(self, "Invalid File", "Please select a valid CSV file (.csv extension).")
        # else: keep previous state

    def run_algorithm(self):

         # --- Reset prediction state whenever a new training run starts ---
        self.trained_model_instance = None
        self.predict_dataset_path = None
        self.upload_predict_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.predict_status_label.setText("No prediction dataset selected.")
        self.predict_status_label.setStyleSheet("font-style: italic; color: #555;")

        selected_display_name = self.algorithm_combo.currentText()
        AlgorithmClass = self.algorithm_map.get(selected_display_name)

        if not AlgorithmClass:
            QMessageBox.critical(self, "Error", f"Algorithm class for '{selected_display_name}' not found!")
            return
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please upload a dataset first.")
            return

        # Disable buttons during run
        self.run_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.algorithm_combo.setEnabled(False)
        QApplication.processEvents() # Update UI

        output_text = ""
        metrics_text = None
        algo_output_dir = None
        error_occurred = False
        algo_instance = None # Define algo_instance here

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                print(f"--- Initializing {selected_display_name} ---")
                # algorithms.py called to create algorithm constructor
                algo_instance = AlgorithmClass(self.dataset_path)
                algo_output_dir = algo_instance.get_output_dir() # Get output dir early

                # Train the model
                print(f"\n--- Training {selected_display_name} ---")
                algo_instance.train()

                # Evaluate the model
                print(f"\n--- Evaluating {selected_display_name} ---")
                algo_instance.evaluate()
                print("\n--- Run Finished ---")

            # Combine stdout and stderr for the results window
            output_text = stdout_capture.getvalue() + "\n--- Errors/Warnings (if any) ---\n" + stderr_capture.getvalue()

            # Try reading the metrics file generated by evaluate()
            metrics_file_path = os.path.join(algo_output_dir, "metrics.txt")
            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, 'r') as f:
                    metrics_text = f.read()
            else:
                print(f"Warning: metrics.txt not found in {algo_output_dir}")
                metrics_text = "Metrics file not found."

            # --- IMPORTANT: Store the trained instance ---
            self.trained_model_instance = algo_instance
            # ---

        except Exception as e:
            error_occurred = True
            print(f"An error occurred during {selected_display_name} execution:")
            traceback.print_exc() # Print full traceback to console
            # Combine stdout/stderr captured so far with the error message
            error_details = f"--- Error during execution ---\n{traceback.format_exc()}"
            output_text = stdout_capture.getvalue() + "\n--- Errors/Warnings ---\n" + stderr_capture.getvalue() + "\n" + error_details
            # Ensure output dir is cleared or handled if error happened mid-way
            algo_output_dir = None # Don't try to show plots if it failed badly


        finally:
            # Re-enable buttons
            self.run_button.setEnabled(True)
            self.upload_button.setEnabled(True)
            self.algorithm_combo.setEnabled(True)
            QApplication.processEvents() # Update UI

            # Show results/error window
            results_win = ResultsWindow(
                output_text,
                parent=self,
                metrics_text=metrics_text,
                selected_algorithm=selected_display_name,
                algorithm_output_dir=algo_output_dir, # Will be None if error occurred early
                is_error=error_occurred
            )
            results_win.exec() # Show modally

            # --- Enable prediction buttons ONLY if run was successful ---
            if not error_occurred and self.trained_model_instance:
                self.upload_predict_button.setEnabled(True)
                # predict_button remains disabled until prediction file is uploaded
                self.predict_status_label.setText("Ready to upload prediction dataset.")
                self.predict_status_label.setStyleSheet("font-style: normal; color: green;")
                print("Algorithm run successful.\nPrediction buttons enabled.")
            else:
                 # Keep prediction buttons disabled
                 self.upload_predict_button.setEnabled(False)
                 self.predict_button.setEnabled(False)
                 self.trained_model_instance = None # Clear instance on error
                 self.predict_status_label.setText("Algorithm run failed or was interrupted.")
                 self.predict_status_label.setStyleSheet("font-style: italic; color: red;")
                 print("Error: Algorithm run failed.\nPrediction buttons remain disabled.")


     # --- NEW: Function to run prediction on new data ---
    def run_prediction(self):
        if self.trained_model_instance is None:
            QMessageBox.warning(self, "Warning", "No model has been trained successfully yet. Please run an algorithm first.")
            return
        if not self.predict_dataset_path:
            QMessageBox.warning(self, "Warning", "Please upload a dataset for prediction first using 'Upload Prediction CSV'.")
            return

        # Disable buttons during prediction
        self.predict_button.setEnabled(False)
        self.upload_predict_button.setEnabled(False)
        QApplication.processEvents() # Update UI

        predictions_class = None
        error_occurred = False
        error_message = ""

        try:
            print(f"\n--- Starting Prediction using {self.trained_model_instance.__class__.__name__} ---")
            model = self.trained_model_instance

            # --- 1. Load New Data ---
            print(f"Loading prediction data from: {self.predict_dataset_path}")
            try:
                new_df = pd.read_csv(self.predict_dataset_path)
            except Exception as e:
                raise ValueError(f"Failed to load prediction CSV: {e}") # Re-raise as ValueError

            # --- 2. Preprocess New Data (CRITICAL: Use scaler from trained model) ---
            print("\nPreprocessing prediction data...\n")
            if not hasattr(model, 'scaler') or model.scaler is None:
                 raise AttributeError("Trained model instance is missing the 'scaler' object.")
            if not hasattr(model, 'df') or model.df is None:
                 raise AttributeError("Trained model instance is missing the original 'df' DataFrame.")

            # Handle missing values (using simple fillna(0) for safety, could be improved)
            # A better approach would be to save the means during training and use them here.
            if new_df.isnull().values.any():
                 print("Warning: Missing values found in prediction data. Filling with 0.")
                 new_df.fillna(0, inplace=True)

            # Feature Consistency Check
            # Get feature columns used during training (assuming target is last col)
            try:
                # model.x might be numpy array after scaling in original run, use original df cols
                training_feature_cols = model.df.columns[:-1]

                # Check if all required columns are present
                missing_cols = [col for col in training_feature_cols if col not in new_df.columns]
                if missing_cols:
                    raise ValueError(f"Prediction data is missing required columns: {missing_cols}")

                # Select only the necessary columns in the correct order
                new_x = new_df[training_feature_cols]

            except Exception as e:
                 raise ValueError(f"Error aligning prediction data columns: {e}")


            # Scale features using the *trained* scaler
            print("Scaling features...\n")
            try:
                new_x_scaled = model.scaler.transform(new_x) 
            except ValueError as e:
                 # More specific error for shape mismatch during scaling
                 n_expected = model.scaler.n_features_in_
                 n_got = new_x.shape[1]
                 raise ValueError(f"Feature mismatch during scaling. Expected {n_expected} features, but got {n_got}. Check columns. Original error: {e}")
            except Exception as e:
                 raise ValueError(f"Error scaling prediction data: {e}")

            # Reshape for LSTM/RNN if necessary
            if isinstance(model, (algorithms.LSTM, algorithms.RNN)):
                print("Reshaping data for RNN/LSTM...\n")
                try:
                    # Get expected shape from the *trained* data structure
                    expected_timesteps = model.x_train.shape[1] # Assumes x_train is stored/accessible
                    expected_features = model.x_train.shape[2]
                    # Check if features match after scaling before reshaping
                    if new_x_scaled.shape[1] != expected_features:
                         raise ValueError(f"Scaled feature count ({new_x_scaled.shape[1]}) doesn't match model's expected features ({expected_features}).")
                    new_x_scaled = np.reshape(new_x_scaled, (new_x_scaled.shape[0], expected_timesteps, expected_features))

                except AttributeError:
                     raise AttributeError("Model instance is missing 'x_train' attribute needed for reshaping reference.")
                except Exception as e:
                     raise ValueError(f"Failed to reshape data for LSTM/RNN: {e}")


            # --- 3. Predict using the loaded model ---
            print("Making predictions...\n")
            # Use the model's predict method which should return (class, probability)
            predictions_class, predictions_prob = model.predict(new_x_scaled)
            print("Prediction finished.")

        except (AttributeError, ValueError, Exception) as e: # Catch specific and general errors
            error_occurred = True
            error_message = f"An error occurred during prediction:\n{e}"
            print(error_message) # Print to console
            traceback.print_exc() # Print full traceback

        finally:
            # Re-enable buttons
            self.predict_button.setEnabled(True)
            self.upload_predict_button.setEnabled(True) # Re-enable upload too
            QApplication.processEvents()

            if error_occurred:
                QMessageBox.critical(self, "Prediction Error", error_message)
            else:
                # 4. Display results in the new window
                pred_window = PredictionWindow(predictions_class, parent=self)
                pred_window.exec()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
