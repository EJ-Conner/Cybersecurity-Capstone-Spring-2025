import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QComboBox, 
                            QPushButton, QApplication, QFileDialog, QLabel,
                            QTextEdit, QDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt6.QtGui import QPixmap, QImage
import algorithms
import os

class ResultsWindow(QDialog):
    def __init__(self, output_text, parent=None,confusion_matrix_path=None, roc_curve_path=None, metrics_path=None):
        super().__init__(parent)
        self.setWindowTitle("Algorithm Results")
        self.setFixedSize(1200, 800)
        
        # Layout
        layout = QVBoxLayout()
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText(output_text)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        # Add widgets to layout
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)

        # Display Confusion Matrix
        if confusion_matrix_path and os.path.exists(confusion_matrix_path):
            self.confusion_label = QLabel()
            pixmap = QPixmap(confusion_matrix_path)
            self.confusion_label.setPixmap(pixmap.scaled(700, 500, Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(QLabel("Confusion Matrix:"))
            layout.addWidget(self.confusion_label)

        # Display ROC Curve
        if roc_curve_path and os.path.exists(roc_curve_path):
            self.roc_label = QLabel()
            pixmap = QPixmap(roc_curve_path)
            self.roc_label.setPixmap(pixmap.scaled(700, 500, Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(QLabel("ROC Curve:"))
            layout.addWidget(self.roc_label)

        # Display Metrics
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_text = f.read()
            metrics_display = QTextEdit()
            metrics_display.setReadOnly(True)
            metrics_display.setText(metrics_text)
            layout.addWidget(QLabel("Metrics:"))
            layout.addWidget(metrics_display)

        layout.addWidget(close_button)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        
        # Window setup
        self.setWindowTitle("Machine Learning GUI")
        self.setFixedSize(400, 300)
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
        import io
        import contextlib
        output = io.StringIO()
        
        # Map combo box selection to algorithm class
        algorithm_map = {
            "Gaussian Naive Bayes": algorithms.GaussianNaiveBayes,
            "Random Forest": algorithms.RandomForest,
            "K-Nearest Neighbors": algorithms.KNN,
            "Support Vector Machine": algorithms.SVM,
            "Logistic Regression": algorithms.logReg,
            "LSTM": algorithms.LSTM,
            "RNN": algorithms.RNN,
            "Autoencoder": algorithms.Autoencoder
        }
        
        selected_algorithm = self.algorithm_combo.currentText()
        algorithm_class = algorithm_map[selected_algorithm]
        
        try:
            # Redirect stdout to capture print statements
            with contextlib.redirect_stdout(output):
                algorithm = algorithm_class(self.dataset_path)
                algorithm.train()
                algorithm.evaluate()
            
            # Get the captured output
            result_text = output.getvalue()
            confusion_matrix_path = "program_output/confusion_matrix.png"
            roc_curve_path = "program_output/roc_curve.png"
            metrics_path = "program_output/metrics.txt"

            # Show results in new window
            results_window = ResultsWindow(result_text, self, confusion_matrix_path, roc_curve_path, metrics_path)
            results_window.exec()
            
        except Exception as e:
            error_text = f"Error running algorithm: {str(e)}"
            results_window = ResultsWindow(error_text, self)
            results_window.exec()
        
        finally:
            output.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
