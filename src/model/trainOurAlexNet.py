
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torchvision.io import read_image
from tinyAlexNet import TinyAlexNet


class CNR_EXT_Dataset(Dataset):
    def __init__(self, labels_file, data_dir):
        with open(labels_file, 'r') as f:
            self.img_labels = f.readlines()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx].split(' ')
        label = int(label)
        img = read_image(f'{self.data_dir}/{img_path}').float()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = Resize((150, 150))(img)
        return img, label


batch_size = 256
train_data = CNR_EXT_Dataset(
    'C:/datasets/LABELS/train.txt', 'C:/datasets/PATCHES/')
val_data = CNR_EXT_Dataset(
    'C:/datasets/LABELS/val.txt', 'C:/datasets/PATCHES/')
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=1024)


def train_loop(dataloader, model, loss_fn, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(f'{batch}/{len(dataloader)}')


def test_loop(dataloader, model, loss_fn, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"{name} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    model = TinyAlexNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(train_dataloader, model, loss_fn, 'Train')
        test_loop(val_dataloader, model, loss_fn, 'Val')
        torch.save(model.state_dict(), f'model_weights_e{t}.pth')
    print("Done!")

    torch.save(model.state_dict(), '../../models/model_weights.pth')
