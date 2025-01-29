import sys
import os
import time
import random
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
import numpy as np

PIXELS = 128

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        # Порядок классов в соответствии с папками на скриншоте
        self.classes = [
            "Объект 1", "Объект 2", "Объект 3", "Объект 4", "random/fail", "Объект 5"
        ]
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose model", os.path.expanduser('~'), "Model files (*.keras *.h5 )")
        if not file_path or not os.path.exists(file_path):
            self.warningText = "Модель не выбрана или не найдена"
            print(self.warningText)
            sys.exit()

        self.model = load_model(file_path)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Распознавание рукописного слова')
        self.setGeometry(100, 100, 800, 600)

        self.button = QPushButton('Выбрать изображение для распознавания', self)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap("empty.png").scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        self.timeLabel = QLabel("")
        self.percentLabel = QLabel("")
        self.outputLabel = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)
        layout.addWidget(self.timeLabel)
        layout.addWidget(self.percentLabel)
        layout.addWidget(self.outputLabel)
        self.setLayout(layout)

        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", os.path.expanduser('~'), "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not imagePath:
            return

        pixmap = QPixmap(imagePath)
        self.imageLabel.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

        startTime = time.time()
        prediction = self.predict_image(imagePath)
        endTime = time.time()

        self.timeLabel.setText(f"Время распознавания: {endTime - startTime:.3f} сек.")

        max_index = np.argmax(prediction[0])
        predicted_class = self.classes[max_index]
        confidence = prediction[0][max_index] * 100

        if confidence <= 20:
            predicted_class = "Рандом"
            confidence = 0
            self.percentLabel.setText("Слово никому не принадлежит")
        else:
            self.percentLabel.setText(f"Слово: {predicted_class}, вероятность: {confidence:.3f}%")

        output_text = "\n".join([f"{self.classes[i]}: {prediction[0][i]*100:.3f}%" for i in range(len(self.classes))])
        self.outputLabel.setText(output_text)

    def preprocess_signature(self, image):
        return image

    def predict_image(self, image_path):
        image = cv2.imread(image_path)

        # Изменяем размер изображения и масштабируем значения пикселей
        resized_image = cv2.resize(image, (PIXELS, PIXELS))
        resized_image = resized_image / 255.0
        image_array = np.expand_dims(resized_image, axis=0)

        return self.model.predict(image_array)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
