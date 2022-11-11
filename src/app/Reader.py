import cv2 as cv
import os

class VideoReader:
    def __init__(self, path):
        self.path = path
        self.cap = cv.VideoCapture(path)
    
    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()

class DirectoryReader:
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.index = 0
    
    def read(self):
        if self.index >= len(self.files):
            return None
        img_path = os.path.join(self.path, self.files[self.index])
        self.index += 1
        return cv.imread(img_path)
    
    def release(self):
        # Nothing to do
        pass

class ImageReader:
    def __init__(self, path):
        self.path = path
        self.index = 0
    
    def read(self):
        if self.index >= 1:
            return None
        self.index += 1
        return cv.imread(self.path)
    
    def release(self):
        # Nothing to do
        pass

class Reader:
    def __init__(self, path):
        if os.path.isdir(path):
            self.m_reader = DirectoryReader(path)
            self.mode = 'directory'
        elif path.endswith('.mp4'):
            self.m_reader = VideoReader(path)
            self.mode = 'video'
        elif path.endswith('.jpg') or path.endswith('.png'):
            self.m_reader = ImageReader(path)
            self.mode = 'image'
    
    def read(self):
        return self.m_reader.read()
    
    def release(self):
        self.m_reader.release()