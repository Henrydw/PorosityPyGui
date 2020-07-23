from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui
import PyQt5
import sys

import numpy as np
import cv2
import qimage2ndarray


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Load .ui file
        uic.loadUi("MainWindow.ui", self)
        self.initUI()

    def initUI(self):
        # Set up signals
        self.button_1.clicked.connect(self.clicked)
        self.button_2.clicked.connect(self.binarise)

    def clicked(self):
        self.data = cv2.imread('Camera_mount.png')[..., ::-1]
        self.qimage = qimage2ndarray.array2qimage(self.data)
        self.setImage()

    def binarise(self):
        self.grey = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        _, self.binImg = cv2.threshold(self.grey, 150, 255, cv2.THRESH_BINARY)
        self.qimage = qimage2ndarray.array2qimage(self.binImg)
        self.setImage()

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.setImage()

    def setImage(self):
        if hasattr(self, 'qimage'):
            self.image_1.setPixmap(QtGui.QPixmap(self.qimage).scaled(self.image_1.size(), PyQt5.QtCore.Qt.KeepAspectRatio, PyQt5.QtCore.Qt.SmoothTransformation))


def window():
    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()

    sys.exit(app.exec_())


window()
