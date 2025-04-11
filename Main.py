
import os
import sys
import traceback
import io
import contextlib
from PyQt6.QtWidgets import (QMainWindow, QScrollArea, QWidget, QVBoxLayout, QComboBox, 
                            QPushButton, QApplication, QFileDialog, QLabel,
                            QTextEdit, QDialog, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis, QLineSeries
from PyQt6.QtGui import QPixmap, QImage

try:
    import algorithms
except ImportError:
    print("Error: algorithms.py not found. Make sure it's in the same directory.")
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
        print(f"Debug: Successfully added plot '{title}' from {plot_path}")
    else:
        print(f"Debug: Plot not found or path not provided for '{title}': {plot_path}")
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
        self.scroll_layout = QVBoxLayout(scroll_area_widget) #Layout for scrollable content
        self.scroll_layout.setSpacing(20) #Create spacing between sections
        self.scroll_layout.setContentsMargins(10, 10, 10, 10) #Margins inside scroll area
        scroll_area.setWidget(scroll_area_widget)
        layout.addWidget(scroll_area, 1) # Add scroll area to the main dialog layout, make it stretch

   
        # --- Algorithm Console Output --- (either error if error flag is true or correct output)
        console_label_text = "<b>Error Details:</b>" if self.is_error else "<b>Algorithm Console Output:</b>"
        results_label = QLabel(console_label_text)
        self.scroll_layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText(output_text)
        #Flexible height
        self.results_text.setMinimumHeight(150)
        self.results_text.setMaximumHeight(300) #Limit max initial height
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

            print(f"Results window checking for plots in: {self.algorithm_output_dir}")
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



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.algorithm_instance = None # To hold the algorithm instance

        # --- Window setup ---
        self.setWindowTitle("Botnet Traffic Analysis")
        self.setMinimumSize(500, 400) # Set minimum size
        self.setMaximumSize(650, 500) # Set maximum size
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
             QLabel#DatasetStatusLabel { /* For dataset label */
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
        layout = QVBoxLayout(container) #Apply layout to container
        layout.setContentsMargins(25, 25, 25, 25) #Generous margins
        layout.setSpacing(15) # Spacing between widget groups

        # --- 1. Algorithm selection ---
        layout.addWidget(QLabel("1. Select Algorithm:"))
        self.algorithm_combo = QComboBox()
        # Algorithm Options
        self.algorithm_options = [
            "Gaussian Naive Bayes", "Random Forest", "K Nearest Neighbors",
            "Support Vector Machine", "Logistic Regression", "LSTM", "RNN", "Autoencoder"
        ]
        self.algorithm_combo.addItems(self.algorithm_options)
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

        layout.addStretch(1) # Add flexible space to push run button down

        # --- 3. Run Analysis ---
        layout.addWidget(QLabel("3. Run Analysis:"))
        self.run_button = QPushButton("Run Selected Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)  # Disabled until dataset is uploaded
        layout.addWidget(self.run_button)


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
                # wrong file type uploaded
                self.dataset_label.setStyleSheet("color: #DC143C; font-weight: bold; font-style: normal;") # Crimson red, bold
                self.run_button.setEnabled(False)
                QMessageBox.warning(self, "Invalid File", "Please select a valid CSV file (.csv extension).")
        # else: keep previous state


    def run_algorithm(self):
        # --- Pre-run Checks ---
        if not self.dataset_path:
            QMessageBox.warning(self, "No Dataset", "Please upload a CSV dataset first.")
            return
        if not self.algorithm_combo.currentText():
             QMessageBox.warning(self, "No Algorithm", "Please select an algorithm from the dropdown.")
             return

        selected_algorithm_name = self.algorithm_combo.currentText()
        # Maps name to class in algorithms.py
        algorithm_map = {
            "Gaussian Naive Bayes": algorithms.GaussianNaiveBayes,
            "Random Forest": algorithms.RandomForest,
            "K Nearest Neighbors": algorithms.KNearestNeighbors,
            "Support Vector Machine": algorithms.SupportVectorMachine,
            "Logistic Regression": algorithms.LogisticRegression,
            "LSTM": algorithms.LSTM,
            "RNN": algorithms.RNN,
            "Autoencoder": algorithms.Autoencoder
        }
        algorithm_class = algorithm_map.get(selected_algorithm_name)

        # obsolete
        '''
        if not algorithm_class:
             # Should not happen if combo box items match map keys, but good practice
             QMessageBox.critical(self, "Internal Error", f"Algorithm class for '{selected_algorithm_name}' not found.")
             return
        '''

        # --- Disable UI and Indicate Running ---
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        self.upload_button.setEnabled(False) 
        self.algorithm_combo.setEnabled(False)
        QApplication.processEvents()

        # --- Setup for Execution ---
        output_buffer = io.StringIO() 
        metrics_text = None
        result_text = ""
        algorithm_output_dir = None 

        try:
            # --- Determine and create the algorithm-specific output directory ---
            base_output_dir = "all_outputs"
            algo_folder_name = "".join(filter(str.isalnum, selected_algorithm_name))
            if not algo_folder_name: # handles extra cases if alg folder name isnt generated correctly
                algo_folder_name = "misc_algo"
            algorithm_output_dir = os.path.join(base_output_dir, f"{selected_algorithm_name.replace(' ', '')}_output")
            os.makedirs(algorithm_output_dir, exist_ok=True)
            print(f"Ensured output directory exists: {algorithm_output_dir}")

            # --- Redirect stdout and Run Algorithm ---
            with contextlib.redirect_stdout(output_buffer):
                print(f"--- Running {selected_algorithm_name} ---")
                print(f"Using dataset: {self.dataset_path}")
                print(f"Output will be saved to: {algorithm_output_dir}")

                # 1. Initialize Algorithm
                # Pass the output dir path during initialization 
                print("\nInitializing algorithm and preparing data...") 
                self.algorithm_instance = algorithm_class(self.dataset_path)
                if hasattr(self.algorithm_instance, 'set_output_dir'):
                     self.algorithm_instance.set_output_dir(algorithm_output_dir)
                elif hasattr(self.algorithm_instance, 'output_dir'):
                     self.algorithm_instance.output_dir = algorithm_output_dir 


                print("Initialization complete.")

                # 2. Train Model
                print("\nTraining model...")
                # Ensure the train method saves learning curves if applicable
                train_history = self.algorithm_instance.train() # Store history if returned
                print("Training finished.")

                # 3. Evaluate Model
                print("\nEvaluating model...")
                # Ensure evaluate method saves CM, ROC, metrics.txt
                evaluation_results = self.algorithm_instance.evaluate()
                print("Evaluation finished.")
                print("\n--- End of Execution ---")

            # --- Prepare results for display ---
            result_text = output_buffer.getvalue() # Get captured console output

            # Attempt to read metrics file generated by evaluate()
            metrics_path_in_specific_dir = os.path.join(algorithm_output_dir, "metrics.txt")
            if os.path.exists(metrics_path_in_specific_dir):
                try:
                    with open(metrics_path_in_specific_dir, 'r') as file:
                        metrics_text = file.read()
                except Exception as e:
                    metrics_text = f"--- Error reading metrics file ---\n{str(e)}"
                    print(f"Warning: Could not read metrics file: {e}") #Log warning
            else:
                metrics_text = None 
                print("Metrics file 'metrics.txt' not found in output directory.")

            # --- Show SUCCESS Results Window ---
            results_window = ResultsWindow(
                output_text=result_text,
                parent=self,
                metrics_text=metrics_text,
                selected_algorithm=selected_algorithm_name,
                algorithm_output_dir=algorithm_output_dir, #Pass the correct directory
                is_error=False
            )
            results_window.exec()

        # --- Specific Error Handling ---
        except ImportError as e:
            error_trace = traceback.format_exc()
            error_text = f"Import Error running {selected_algorithm_name}:\n{str(e)}\n\n" \
                         f"Please ensure required libraries (e.g., scikit-learn, tensorflow/keras) are installed.\n\nTraceback:\n{error_trace}"
            print(f"--- ERROR --- \n{error_text}") # Log detailed error
            QMessageBox.critical(self, "Import Error", f"Missing library for {selected_algorithm_name}: {e}. See console/results for details.")
            results_window = ResultsWindow(output_buffer.getvalue() + "\n\n" + error_text, self, selected_algorithm=selected_algorithm_name, is_error=True)
            results_window.exec()

        except FileNotFoundError as e:
            error_trace = traceback.format_exc()
            error_text = f"File Not Found Error running {selected_algorithm_name}:\n{str(e)}\n\n" \
                         f"Ensure the dataset file or another required file exists at the expected path.\n\nTraceback:\n{error_trace}"
            print(f"--- ERROR --- \n{error_text}")
            QMessageBox.critical(self, "File Not Found", f"File not found during {selected_algorithm_name} execution: {e}. See console/results.")
            results_window = ResultsWindow(output_buffer.getvalue() + "\n\n" + error_text, self, selected_algorithm=selected_algorithm_name, is_error=True)
            results_window.exec()

        except ValueError as e: # Catch data-related errors
            error_trace = traceback.format_exc()
            error_text = f"Data Error running {selected_algorithm_name}:\n{str(e)}\n\n" \
                         f"This might be due to incorrect data format, NaN values, or incompatible data types in your CSV.\n\nTraceback:\n{error_trace}"
            print(f"--- ERROR --- \n{error_text}")
            QMessageBox.critical(self, "Data Error", f"Data processing error for {selected_algorithm_name}: {e}. Check data and console/results.")
            results_window = ResultsWindow(output_buffer.getvalue() + "\n\n" + error_text, self, selected_algorithm=selected_algorithm_name, is_error=True)
            results_window.exec()

        except OSError as e: # Catch directory creation errors specifically
            error_trace = traceback.format_exc()
            error_text = f"OS Error (e.g., creating directory) running {selected_algorithm_name}:\n{str(e)}\n\nTraceback:\n{error_trace}"
            print(f"--- ERROR --- \n{error_text}")
            QMessageBox.critical(self, "File System Error", f"Could not create output directory or access file: {e}. Check permissions and path.")
            # Show error in a simple results window without assuming output dir exists
            results_window = ResultsWindow(output_buffer.getvalue() + "\n\n" + error_text, self, selected_algorithm=selected_algorithm_name, is_error=True)
            results_window.exec()

        except Exception as e: # Catch any other unexpected errors
            error_trace = traceback.format_exc()
            error_text = f"An unexpected error occurred while running {selected_algorithm_name}:\n\n{str(e)}\n\nTraceback:\n{error_trace}"
            print(f"--- UNEXPECTED ERROR --- \n{error_text}") # Log detailed error
            QMessageBox.critical(self, "Runtime Error", f"An unexpected error occurred: {e}. Check console/results for details.")
            # Show error, potentially without plots if directory wasn't confirmed
            results_window = ResultsWindow(output_buffer.getvalue() + "\n\n" + error_text, self, selected_algorithm=selected_algorithm_name, algorithm_output_dir=algorithm_output_dir, is_error=True)
            results_window.exec()

        # --- Cleanup ---
        finally:
            output_buffer.close() # Close the string buffer
            # Re-enable UI elements
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Selected Algorithm")
            self.upload_button.setEnabled(True)
            self.algorithm_combo.setEnabled(True)
            self.algorithm_instance = None # Clear instance reference
            QApplication.processEvents() # Ensure UI updates

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
