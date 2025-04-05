import os
import traceback
import sys
import io
import contextlib
from PyQt6.QtWidgets import (QMainWindow, QScrollArea, QWidget, QVBoxLayout, QComboBox, 
                            QPushButton, QApplication, QFileDialog, QLabel,
                            QTextEdit, QDialog, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis, QLineSeries
from PyQt6.QtGui import QPixmap, QImage
import algorithms


class ResultsWindow(QDialog):
    def __init__(self, output_text, parent=None, metrics_text=None,
                 selected_algorithm=None, algorithm_output_dir=None):
        super().__init__(parent)
        self.setWindowTitle(f"{selected_algorithm} Results" if selected_algorithm else "Results")
        self.setMinimumSize(1000, 800)
        self.setLayout(QVBoxLayout())

        # Store passed info
        self.selected_algorithm = selected_algorithm
        self.algorithm_output_dir = algorithm_output_dir


        # Main scrollable area
        scroll_area = QScrollArea()
        scroll_area_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_area_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_area_widget)
        self.layout().addWidget(scroll_area)


        # Results display
        results_label = QLabel("Algorithm Console Output:")
        self.scroll_layout.addWidget(results_label)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText(output_text)
        self.results_text.setMinimumHeight(150)
        self.results_text.setSizePolicy(self.results_text.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        self.scroll_layout.addWidget(self.results_text)

        # --- Metrics display ---
        if metrics_text:
            metrics_label_title = QLabel("Metrics:")
            self.scroll_layout.addWidget(metrics_label_title)
            metrics_display = QTextEdit()
            metrics_display.setReadOnly(True)
            metrics_display.setText(metrics_text)
            metrics_display.setFixedHeight(150)
            self.scroll_layout.addWidget(metrics_display)

        if self.algorithm_output_dir:
            confusion_matrix_path = os.path.join(self.algorithm_output_dir, "confusion_matrix.png")
            roc_curve_path = os.path.join(self.algorithm_output_dir, "roc_curve.png")
            accuracy_curve_path = os.path.join(self.algorithm_output_dir, "learning_curve_accuracy.png")
            loss_curve_path = os.path.join(self.algorithm_output_dir, "learning_curve_loss.png")


            # Confusion Matrix
            if os.path.exists(confusion_matrix_path):
                confusion_label_title = QLabel("Confusion Matrix:")
                self.scroll_layout.addWidget(confusion_label_title)
                confusion_label = QLabel()
                pixmap = QPixmap(confusion_matrix_path)
                confusion_label.setPixmap(pixmap.scaled(600, 500, Qt.AspectRatioMode.KeepAspectRatio))
                self.scroll_layout.addWidget(confusion_label)
            else:
                print(f"Debug: CM plot not found at {confusion_matrix_path}")

               # ROC Curve
            if os.path.exists(roc_curve_path):
                roc_label_title = QLabel("ROC Curve:")
                self.scroll_layout.addWidget(roc_label_title)
                roc_label = QLabel()
                pixmap = QPixmap(roc_curve_path)
                roc_label.setPixmap(pixmap.scaled(600, 500, Qt.AspectRatioMode.KeepAspectRatio))
                self.scroll_layout.addWidget(roc_label)
            else:
                print(f"Debug: ROC plot not found at {roc_curve_path}")

            # --- Learning Curves (Conditional based on algorithm name) ---
            learning_curve_algorithms = ["LSTM", "RNN", "Autoencoder"]
            if self.selected_algorithm in learning_curve_algorithms:
                # Accuracy Curve
                if os.path.exists(accuracy_curve_path):
                    lc_acc_label_title = QLabel("Learning Curve (Accuracy):")
                    self.scroll_layout.addWidget(lc_acc_label_title)
                    lc_acc_label = QLabel()
                    acc_pixmap = QPixmap(accuracy_curve_path)
                    lc_acc_label.setPixmap(acc_pixmap.scaled(600, 500, Qt.AspectRatioMode.KeepAspectRatio))
                    self.scroll_layout.addWidget(lc_acc_label)
                else:
                     print(f"Debug: Accuracy curve not found at {accuracy_curve_path}")

                # Loss Curve
                if os.path.exists(loss_curve_path):
                    lc_loss_label_title = QLabel("Learning Curve (Loss):")
                    self.scroll_layout.addWidget(lc_loss_label_title)
                    lc_loss_label = QLabel()
                    loss_pixmap = QPixmap(loss_curve_path)
                    lc_loss_label.setPixmap(loss_pixmap.scaled(600, 500, Qt.AspectRatioMode.KeepAspectRatio))
                    self.scroll_layout.addWidget(lc_loss_label)
                else:
                     print(f"Debug: Loss curve not found at {loss_curve_path}")

            # --- Close button ---
            
            #self.scroll_layout.addStretch() # Optional: uncomment if you want button at bottom
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close)
            self.scroll_layout.addWidget(close_button)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        
        # Window setup
        self.setWindowTitle("Machine Learning GUI")
        self.setFixedSize(500, 400)
        self.setStyleSheet(
            "font-family: arial; "
            "color: black; "
            "font-size: 16px; "
            "background-color: #f0f0f0;"
        )
        
        # Main widget and layout
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout()
        
        # Algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Gaussian Naive Bayes",
            "Random Forest",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Logistic Regression",
            "LSTM",
            "RNN",
            "Autoencoder"
        ])
        
        # Dataset upload button
        self.upload_button = QPushButton("Upload Dataset (CSV)")
        self.upload_button.clicked.connect(self.upload_dataset)
        
        # Dataset status label
        self.dataset_label = QLabel("No dataset uploaded")
        self.dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Run button
        self.run_button = QPushButton("Run Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)  # Disabled until dataset is uploaded
        
        # Format buttons
        for button in (self.upload_button, self.run_button):
            button.setStyleSheet(
                "border-radius: 8px; "
                "background-color: #4CAF50; "
                "color: white; "
                "padding: 8px;"
            )
        
        # Add widgets to layout
        layout.addWidget(QLabel("Select Algorithm:"))
        layout.addWidget(self.algorithm_combo)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.dataset_label)
        layout.addStretch()
        layout.addWidget(self.run_button)
        
        container.setLayout(layout)
    
    def upload_dataset(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Dataset",
            "",
            "CSV Files (*.csv)"
        )
        
        if file_name:
            if file_name.endswith('.csv'):
                self.dataset_path = file_name
                self.dataset_label.setText(f"Dataset: {file_name.split('/')[-1]}")
                self.run_button.setEnabled(True)
            else:
                self.dataset_label.setText("Please upload a CSV file")
    
    def run_algorithm(self):
        if not self.dataset_path:
            return
        
        # Capture print output
        output = io.StringIO()
        final_metrics = {}
        #final_metrics = algorithm.evaluate()
        selected_algorithm = self.algorithm_combo.currentText()
       # training_history = None  # Initialize training history
        
            # --- Determine and create the algorithm-specific output directory ---
        base_output_dir = "all_outputs"
        algo_folder_name = "".join(filter(str.isalnum, selected_algorithm))
        algorithm_output_dir = os.path.join(base_output_dir, f"{algo_folder_name}_output")
        try:
            os.makedirs(algorithm_output_dir, exist_ok=True)
            print(f"Ensured output directory exists: {algorithm_output_dir}") # Debug print
        except OSError as e:
            print(f"Error creating directory {algorithm_output_dir}: {e}")
            return
        
        # Map combo box selection to algorithm class
        algorithm_map = {
            "Gaussian Naive Bayes": algorithms.GaussianNaiveBayes,
            "Random Forest": algorithms.RandomForest,
            "K-Nearest Neighbors": algorithms.KNearestNeighbors,
            "Support Vector Machine": algorithms.SupportVectorMachine,
            "Logistic Regression": algorithms.LogisticRegression,
            "LSTM": algorithms.LSTM,
            "RNN": algorithms.RNN,
            "Autoencoder": algorithms.Autoencoder
        }
        
     
        algorithm_class = algorithm_map[selected_algorithm]

        try:
        # Redirect stdout to capture print statements
            with contextlib.redirect_stdout(output):
                print(f"Running {selected_algorithm}...")
                # Algorithm class now knows its output dir via get_output_dir()
                algorithm = algorithm_class(self.dataset_path)
                print("Training model...")
                algorithm.train() # Train might save learning curves
                #QApplication.processEvents()
                print("Evaluating model...")
                final_metrics = algorithm.evaluate() # Evaluate saves CM, ROC and returns metrics
                #QApplication.processEvents()

        # Get captured print output
                result_text = output.getvalue()
                confusion_matrix_path_in_specific_dir = os.path.join(algorithm_output_dir, "confusion_matrix.png")
                roc_curve_path_in_specific_dir = os.path.join(algorithm_output_dir, "roc_curve.png")
                metrics_path_in_specific_dir = os.path.join(algorithm_output_dir, "metrics.txt")
              
                with open(metrics_path_in_specific_dir, 'r') as file:
                    metrics_text = file.read()
                
                
                #################################### - Learning Curve?
                
                results_window = ResultsWindow(
                    output_text=result_text,
                    parent=self,
                    metrics_text=metrics_text,
                    selected_algorithm=selected_algorithm, 
                    algorithm_output_dir=algorithm_output_dir
                )
        
                results_window.exec()

        except Exception as e:
            error_trace = traceback.format_exc()
            error_text = f"Error running {selected_algorithm}:\n{str(e)}\n\nTraceback:\n{error_trace}"
            # Show error in a results window (without plots)
            results_window = ResultsWindow(error_text, self)
            results_window.exec()
        
            
        finally:
            output.close()
            
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
