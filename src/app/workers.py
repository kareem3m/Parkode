from app.Transform import transform
from app.Reader import Reader

from PySide6.QtCore import Signal, QObject, Slot, QTimer
import torch, time, cv2 as cv
from app.violationDetector import ViolationDetector
from app.PlateDetection import plateDetector

import sys


class OccupancyWorker(QObject):
    occupancyUpdated = Signal(dict)

    def __init__(self):
        super(OccupancyWorker, self).__init__()
        self.active = False
        self.occupancyStatus = {}

    @Slot()
    def set_slots(self, slots):
        self.slots = slots
        self.occupancyStatus = {}
        print("OccupancyWorker: slots set")

    @Slot()
    def set_model(self, model):
        self.model = model
        print("OccupancyWorker: model set")

    @Slot()
    def processFrame(self, frame):
        if self.active:
            for slot_id in self.slots:
                x1, y1, x2, y2 = self.slots[slot_id]
                x = transform(frame[y1:y2, x1:x2])
                prediction = self.model.predict(x)
                self.occupancyStatus[slot_id] = prediction
            self.occupancyUpdated.emit(self.occupancyStatus)

    @Slot()
    def disable(self):
        self.active = False

    @Slot()
    def enable(self):
        self.active = True

    @Slot()
    def setState(self, state):
        if state == 2:
            self.active = True
        else:
            self.active = False
            self.occupancyStatus = {}
            self.occupancyUpdated.emit(self.occupancyStatus)
        print(f"OccupancyWorker: state set = {self.active}")


class ViolationsWorker(QObject):
    overlapsUpdated = Signal(list)
    foreignsUpdated = Signal(list)

    def __init__(self) -> None:
        super(ViolationsWorker, self).__init__()
        self.active = False
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        self.overlaps = []
        self.foreigns = []
        self.ready = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setReady)
        self.timer.setInterval(1200)
        self.timer.start()

    @Slot()
    def set_slots(self, slots):
        self.slots = slots
        self.detector = ViolationDetector(self.slots, self.yolo.names)
        self.overlaps = []
        self.foreigns = []
        print("ViolationsWorker: slots set")

    @Slot()
    def processFrame(self, frame):
        if self.active and self.ready:
            self.ready = False
            st = time.time()
            det = self.yolo(frame).xyxy[0]
            self.overlaps, self.foreigns = self.detector.detect(det)
            self.overlapsUpdated.emit(self.overlaps)
            self.foreignsUpdated.emit(self.foreigns)
            # print execution time in ms
            print(
                f"ViolationDetector: detect() took {(time.time() - st) * 1000} ms")

    @Slot()
    def setReady(self):
        self.ready = True

    @Slot()
    def disable(self):
        self.active = False

    @Slot()
    def enable(self):
        self.active = True

    @Slot()
    def setState(self, state):
        if state == 2:
            self.active = True
        else:
            self.active = False
            self.overlaps = []
            self.foreigns = []
            self.overlapsUpdated.emit(self.overlaps)
            self.foreignsUpdated.emit(self.foreigns)
        print(f"ViolationsWorker: state set = {self.active}")


class ANPRWorker(QObject):
    platesUpdated = Signal(dict)

    def __init__(self):
        super(ANPRWorker, self).__init__()
        self.active = False
        self.plates = {}

    @Slot()
    def set_slots(self, slots):
        self.slots = slots
        self.plates = {}
        print("ANPRWorker: slots set")

    @Slot()
    def processFrame(self, frame):
        if self.active:
            print("ANPRWorker: processing frame with shape ", frame.shape)
            try:
                for slot_id in self.slots:
                    x1, y1, x2, y2 = self.slots[slot_id]
                    plate = plateDetector.detect(frame[y1:y2, x1:x2])
                    self.plates[slot_id] = plate
            except Exception as _:
                print("Call Me..")
            self.platesUpdated.emit(self.plates)
            print("plates: ", self.plates)

    @Slot()
    def disable(self):
        self.active = False

    @Slot()
    def enable(self):
        self.active = True

    @Slot()
    def setState(self, state):
        if state == 2:
            self.active = True
        else:
            self.active = False
            self.plates = {}
            self.platesUpdated.emit(self.plates)
        print(f"ANPRWorker: state set = {self.active}")


class ReaderWorker(QObject):
    frameUpdated = Signal(cv.Mat)

    def __init__(self):
        super(ReaderWorker, self).__init__()
        self.active = False
        self.frame = None
        self.reader = None

    @Slot()
    def set_reader(self, mediaPath):
        self.reader = Reader(mediaPath)

    def readFrame(self):
        if self.active and self.reader is not None:
            self.frame = self.reader.read()
            if self.frame is not None:
                self.frameUpdated.emit(self.frame)

    @Slot()
    def run(self):
        self.active = True
        fps = 15
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.readFrame)
        self.timer.setInterval(1000 / fps)
        self.timer.start()

    @Slot()
    def disable(self):
        self.active = False
        self.timer.stop()
