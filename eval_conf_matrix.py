""" the confusion matrix of the MNIST dataset of the trained neural net (with
best hyperparameters) is calculated and plotted. """
import time
from typing import Any
from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from model import Network
from utils.plotcm import plot_confusion_matrix
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

start_time = time.time()


# add more analytical tools
# predictions (on cpu):

network = Network()
network.load_state_dict(torch.load('best_result/SavedModel.pth'))  # loads the saved models weights etc.


def get_num_correct(preds: torch.Tensor, labels: torch.Tensor) -> Union[int, float, bool]:
    return preds.argmax(dim=1).eq(labels).sum().item()


train_set = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


def get_all_preds(model: Network, loader: Any) -> torch.Tensor:
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

preds_correct = get_num_correct(train_preds, train_set.targets)

print(f'correctly predicted images: {preds_correct} of {len(train_set)}')
print(f'accuracy of prediction: {round((preds_correct / len(train_set))*100, 2)} %')

stacked = torch.stack((train_set.targets, train_preds.argmax(dim=1)), dim=1)
cmt = torch.zeros(10, 10, dtype=torch.int32)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] += 1
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
names = tuple(train_set.classes)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, names)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
