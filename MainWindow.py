import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui, QtCore
import PyQt5
import numpy as np
import cv2
import qimage2ndarray
import pyqtgraph as pg

from Porosity import findSectionMask, porosityBasic


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Set default pyqtgraph background color
        pg.setConfigOption('background', 'w')
        # Load .ui file
        uic.loadUi("MainWindow.ui", self)
        self.initUI()

    def initUI(self):
        # set up thread
        self.thread = worker_porosityBasic()
        self.thread.output[np.ndarray, np.ndarray].connect(self.plot_res)
        self.thread.finished.connect(self.re_enable_proc)

        # Set up signals
        self.button_1.clicked.connect(self.clicked)
        self.button_2.clicked.connect(self.basicP)

    def clicked(self):
        self.data = cv2.imread('example.jpeg')[..., ::-1]
        self.qimage = qimage2ndarray.array2qimage(self.data)
        self.setImage()

    def basicP(self):

        self.button_2.setEnabled(False)
        # -------------- prep for output --------------
        # EG: make a blank array etc
        # -------------- start the thread --------------
        self.thread.proc(self.data)

    def plot_res(self, porosityVals, thresholdVals):
        self.graphWidget.clear()
        PlotItem = self.graphWidget.getPlotItem()
        # PlotItem.setAspectLocked(lock=True)
        self.graphWidget.addLegend()  # ?must come before first call to plot.

        self.graphWidget.setXRange(min(thresholdVals), max(thresholdVals), padding=0)
        self.graphWidget.setYRange(0, 1, padding=0)

        pen = pg.mkPen(color='#f97306', width=2, style=QtCore.Qt.SolidLine)
        self.graphWidget.addLine(x=1, name="Threshold")
        self.graphWidget.plot(thresholdVals, 1 - porosityVals, pen=pen, name="Porosity")

    def re_enable_proc(self):
        self.button_2.setEnabled(True)

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.setImage()

    def setImage(self):
        if hasattr(self, 'qimage'):
            self.image_1.setPixmap(QtGui.QPixmap(self.qimage).scaled(self.image_1.size(), PyQt5.QtCore.Qt.KeepAspectRatio, PyQt5.QtCore.Qt.SmoothTransformation))


# -------------------------------- New Thread --------------------------------
class worker_porosityBasic(QtCore.QThread):
    # -------------- a signal to be emitted --------------
    output = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.exiting = False
        self.img = 0

    def __del__(self):
        self.exiting = True  # this call links to run method and stops infinte loop and waits for the last round to finish
        self.wait()

    def proc(self, img):
        # -------------- accept input and start processing --------------
        self.img = img
        self.start()

    def run(self):
        # Note: This is never called directly. It is called by Qt once the
        # thread environment has been set up.

        # Image processing
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        binImg = cv2.threshold(gray, int(100), 255, cv2.THRESH_BINARY)[1]
        masks = findSectionMask(binImg, area='hatch')
        res = np.multiply(gray, masks[0])

        # Finding porosity
        porosityVals, thresholdVals = porosityBasic(gray, masks[0], start=0, end=255)

        # emit the results of each loop
        self.output.emit(porosityVals, thresholdVals)


def window():

    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()

    sys.exit(app.exec_())


window()
