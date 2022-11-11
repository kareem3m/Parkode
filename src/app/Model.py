import torch
import sys
from model.tinyAlexNet import TinyAlexNet
from joblib import load



class SVMModel():
    def __init__(self, filename):
        self.clf = load(filename)

    def predict(self, x):
        return self.clf.predict(x.reshape(1, -1))


class CNNModel():
    def __init__(self, filename):
        self.model = TinyAlexNet()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(self, x):
        return self.model(x).argmax().item()


class Model:
    def __init__(self, type, filename):
        self.type = type
        self.weights = filename
        if type == 'svm':
            self.m_model = SVMModel(filename)
        elif type == 'cnn':
            self.m_model = CNNModel(filename)

    def predict(self, x):
        return self.m_model.predict(x)
