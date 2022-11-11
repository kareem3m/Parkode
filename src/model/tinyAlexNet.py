import torch
from torch.nn import Conv2d, Linear
from torch.nn.functional import relu, max_pool2d, softmax

class TinyAlexNet(torch.nn.Module):
    def __init__(self):
        super(TinyAlexNet, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=11, stride=4, groups=1, bias=True)
        self.conv2 = Conv2d(16, 8, kernel_size=5, stride=2, groups=1, bias=True)
        self.fc4 = Linear(72, 2, bias=True)

    def forward(self, x):
        # input 150x150x3
        conv1 = self.conv1(x)
        # conv1 35x35x16
        relu1 = relu(conv1)
        pool1 = max_pool2d(relu1, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        # pool1 17x17x16
        conv2 = self.conv2(pool1)
        # conv2 7x7x8
        relu2 = relu(conv2)
        pool2 = max_pool2d(relu2, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        # pool2 3x3x8
        fc4 = self.fc4(torch.flatten(pool2, 1))
        # fc4 2
        relu4 = relu(fc4)
        score = softmax(relu4, dim=1)
        return score
        