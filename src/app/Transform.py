import cv2 as cv
from torchvision.transforms import ToTensor, Resize

def transform(x):
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        tensor = ToTensor()(x)
        tensor *= 255
        tensor = Resize((150, 150))(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor