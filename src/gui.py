# This Python file uses the following encoding: utf-8
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from app.Model import Model
from app.helpers.slots_readers import read_slots
import cv2 as cv
from PySide6.QtGui import QIcon, QImage, QPixmap, QFont
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QFileSystemModel
import os
from pathlib import Path
import sys
from app.workers import ReaderWorker, OccupancyWorker, ViolationsWorker, ANPRWorker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class MainWindow(QMainWindow):
    slotsChanged = Signal(dict)
    transformChanged = Signal(str)
    modelChanged = Signal(Model)
    mediaPathChanged = Signal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.load_ui()
        self.setWindowTitle("Parkode")
        self.setWindowIcon(QIcon('logo.png'))

        self.occupancyStatus = {}
        # Creating occupancy worker
        self.occupancyWorker = OccupancyWorker()
        self.slotsChanged.connect(
            self.occupancyWorker.set_slots, Qt.QueuedConnection)
        self.modelChanged.connect(
            self.occupancyWorker.set_model, Qt.QueuedConnection)
        self.occupancyWorker.occupancyUpdated.connect(
            self.updateOccupancyStatus, Qt.QueuedConnection)

        self.overlaps = []
        self.foreigns = []
        # Creating violations worker
        self.violationsWorker = ViolationsWorker()
        self.slotsChanged.connect(
            self.violationsWorker.set_slots, Qt.QueuedConnection)
        self.violationsWorker.overlapsUpdated.connect(
            self.updateOverlaps, Qt.QueuedConnection)
        self.violationsWorker.foreignsUpdated.connect(
            self.updateForeigns, Qt.QueuedConnection)

        self.plates = {}
        # Creating ANPR worker
        self.anprWorker = ANPRWorker()
        self.slotsChanged.connect(
            self.anprWorker.set_slots, Qt.QueuedConnection)
        self.anprWorker.platesUpdated.connect(
            self.updatePlates, Qt.QueuedConnection)

        # Creating reader worker
        self.readerWorker = ReaderWorker()
        self.mediaPathChanged.connect(
            self.readerWorker.set_reader, Qt.QueuedConnection)
        self.readerWorker.frameUpdated.connect(
            self.occupancyWorker.processFrame, Qt.QueuedConnection)
        self.readerWorker.frameUpdated.connect(
            self.violationsWorker.processFrame, Qt.QueuedConnection)
        self.readerWorker.frameUpdated.connect(
            self.updateFrame, Qt.QueuedConnection)
        self.readerWorker.frameUpdated.connect(
            self.anprWorker.processFrame, Qt.QueuedConnection)

        self.readModel(os.path.join(os.path.dirname(__file__), '..\models\cnn_weights.pth'))

        # Creating Threads
        self.occupancyThread = QThread()
        self.readerThread = QThread()
        self.violationsThread = QThread()
        self.anprThread = QThread()

        # Connecting threads signals
        self.occupancyThread.finished.connect(self.occupancyWorker.disable)
        self.occupancyThread.finished.connect(self.occupancyWorker.deleteLater)
        self.violationsThread.finished.connect(self.violationsWorker.disable)
        self.violationsThread.finished.connect(
            self.violationsWorker.deleteLater)
        self.anprThread.finished.connect(self.anprWorker.disable)
        self.anprThread.finished.connect(self.anprWorker.deleteLater)
        self.readerThread.finished.connect(self.readerWorker.disable)
        self.readerThread.finished.connect(self.readerWorker.deleteLater)
        self.readerThread.started.connect(self.readerWorker.run)

        # Starting threads
        self.occupancyWorker.moveToThread(self.occupancyThread)
        self.occupancyThread.start()
        self.readerWorker.moveToThread(self.readerThread)
        self.readerThread.start()
        self.violationsWorker.moveToThread(self.violationsThread)
        self.violationsThread.start()
        self.anprWorker.moveToThread(self.anprThread)
        self.anprThread.start()

        self.label = self.win.label
        self.occupancyCheckBox = self.win.occupancyCheckBox
        self.violationsCheckBox = self.win.violationsCheckBox
        self.anbrCheckBox = self.win.ANPRCheckBox
        self.occupancyCheckBox.stateChanged.connect(
            self.occupancyWorker.setState)
        self.violationsCheckBox.stateChanged.connect(
            self.violationsWorker.setState)
        self.anbrCheckBox.stateChanged.connect(self.anprWorker.setState)

        self.loadButton = self.win.loadButton
        self.loadButton.clicked.connect(self.setMediaPath, Qt.QueuedConnection)

        self.fileList = self.win.fileList
        self.fileModel = QFileSystemModel()
        self.fileModel.setRootPath(os.path.join(os.path.dirname(__file__), '../data_samples'))
        self.fileList.setModel(self.fileModel)
        self.fileList.setRootIndex(self.fileModel.index(os.path.join(os.path.dirname(__file__), '../data_samples')))
        # filter media files
        self.fileModel.setNameFilters(["*.mp4", "*.jpg", "*.png"])
        self.fileModel.setNameFilterDisables(False)

        self.log = self.win.log
        self.log.setReadOnly(True)
        # setting font size for log
        font = QFont()
        font.setPointSize(14)
        self.log.setFont(font)

        self.win.searchButton.clicked.connect(self.searchPlate)

        self.mediaPath = None

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile(os.path.join(os.path.dirname(__file__), 'app/form.ui'))
        ui_file.open(QFile.ReadOnly)
        self.win = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.win)

    @Slot(dict)
    def updateOccupancyStatus(self, status):
        self.occupancyStatus = status

    @Slot(list)
    def updateOverlaps(self, overlaps):
        self.overlaps = overlaps

    @Slot(list)
    def updateForeigns(self, foreigns):
        self.foreigns = foreigns

    @Slot(dict)
    def updatePlates(self, plates):
        self.plates = plates
        self.updateFrame(self.frame)
        print(self.plates)

    @Slot()
    def searchPlate(self):
        plate = self.win.plateNumber.toPlainText()
        self.log.appendPlainText("Searching for plate number: " + plate)
        if plate in self.plates.values():
            self.log.appendPlainText("Plate number found: " + plate)
            keys = list(self.plates.keys())
            values = list(self.plates.values())
            pos = str(keys[values.index(plate)])
            self.log.appendPlainText(
                "Plate number: " + plate + " at slot " + pos)
        else:
            self.log.appendPlainText("Plate number not found: " + plate)

    @Slot()
    def updateFrame(self, frame):
        self.frame = frame
        frame = frame.copy()
        for slot_id, occupied in self.occupancyStatus.items():
            x1, y1, x2, y2 = self.slots[slot_id]
            cv.putText(frame, str(slot_id), (x1, y1),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if occupied:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
            else:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for overlap in self.overlaps:
            x1, y1, x2, y2 = self.slots[overlap]
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            self.log.appendPlainText(
                f"Space overlap detected at slot {overlap}")

        for foreign in self.foreigns:
            vehicle_type = foreign[0]
            for slot in foreign[1]:
                x1, y1, x2, y2 = self.slots[slot]
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(frame, vehicle_type, (x1, y1),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.log.appendPlainText(
                    f'Foreign vehicle detected: {vehicle_type} in slot {slot}\n')

        for slot_id in self.plates:
            x1, y1, x2, y2 = self.slots[slot_id]
            font = ImageFont.truetype('./arialbd.ttf', 300)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 255))
            w, h = draw.textsize(self.plates[slot_id], font=font)
            draw.rectangle((x1, y1, x1+int(1.1*w), y1 +
                           int(1.1*h)), fill=(14, 201, 255))
            draw.text((x1, y1), self.plates[slot_id],
                      font=font, fill=(0, 0, 0))
            frame = np.array(img_pil)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        print('closing...')
        self.occupancyThread.quit()
        self.readerThread.quit()
        return super(MainWindow, self).closeEvent(event)

    def readSlots(self, filename):
        print("reading slots from file: ", filename)
        self.slots = {}
        try:
            slots = read_slots(filename)
            for slot_id, x1, y1, x2, y2 in slots:
                self.slots[slot_id] = (x1, y1, x2, y2)
            self.slotsChanged.emit(self.slots)
        except Exception as e:
            print("Slots not found: ", e)

    def readModel(self, filename):
        self.model = Model('cnn', filename)
        self.modelChanged.emit(self.model)

    @Slot()
    def setMediaPath(self):
        # get selected file in the file list
        index = self.fileList.currentIndex()
        path = self.fileModel.filePath(index)
        if path != self.mediaPath:
            self.mediaPath = path
            self.occupancyStatus = {}
            self.overlaps = []
            self.foreigns = []
            self.plates = {}
            print("media path: ", self.mediaPath)
            # get parent directory of the selected file
            dirname = os.path.dirname(path)
            # search for the csv file in the parent directory
            for filename in os.listdir(dirname):
                if filename.endswith(".csv"):
                    print("found csv file: ", filename)
                    self.readSlots(dirname + "/" + filename)
                    break
            else:
                print("No slots file found")
        self.mediaPathChanged.emit(self.mediaPath)


if __name__ == "__main__":
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
