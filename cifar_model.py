import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CustomImageDataset(Dataset):

    def __init__(self, file, transform=None, target_transform=None):
        self.unpickled_file = unpickle(file)
        self.img_labels = self.unpickled_file[b'labels']
        self.img_dir = self.unpickled_file[b'filenames']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        numpy_array_of_data = self.__dict__['unpickled_file'][b'data'][idx]
        tensor_of_image = np.asarray(numpy_array_of_data, dtype=np.float32)
        label = self.img_labels[idx]
        return tensor_of_image, label


data_set = CustomImageDataset(r'C:\Users\leoma\PycharmProjects\PyTorch\MORE DATASETS\DataBatches\AccessibleBatches\data_batch_1')

train_ds, val_ds = torch.utils.data.random_split(data_set, [9000, 1000])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)

input_size = 32*32*3
num_classes = 10


def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(torch.eq(preds, labels)).item() / len(preds))


def evaluate(model, val_loader):
    output = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(output)


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


class Output:
    def __init__(self, training_data, model):
        for image, label in training_data:
            self.output = model(image)
            break


img_sort_model = MnistModel()

output = Output(train_loader, img_sort_model)

probs = F.softmax(output.output, dim=1)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []  # for recording epoch-wise results

    for epoch in range(epochs):

        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

if __name__ == "__main__":
    history1 = fit(20, 0.1, img_sort_model, train_loader, val_loader)
