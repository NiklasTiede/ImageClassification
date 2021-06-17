""" Training script to train the weights of the
CNN Network for image recognition. """
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Network
from utils.run_tools import RunBuilder
from utils.run_tools import RunManager
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

start_time = time.time()


# how to use only GPU for training?  -> still to be implemented
# use_gpu = True if torch.cuda.is_available() else False  # rest of the code?

# splitting the classes into another file ! getting a good structure and add docstrings and annotations to everything !


train_set = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


params = OrderedDict(
    lr=[.01],
    batch_size=[1000],
    shuffle=[False],
    # input_chan=[1],
    # conv1out_conv2in=[6],
    # kernsize_conv1=[5],
    # conv2out_fc1in=[12],
    # kernsize_conv2=[5],
    # fc1out_fc2in=[120],
    # fc2out_outin=[60],
    # out_feat=[10]
)


b = RunBuilder()
m = RunManager()

b.get_runs(params)

for run in RunBuilder.get_runs(params):
    print('print run:', type(run))
    # network = Network(input_chan=run.input_chan, conv1out_conv2in=run.conv1out_conv2in, kernsize_conv1=run.kernsize_conv1,
    #                   conv2out_fc1in=run.conv2out_fc1in, kernsize_conv2=run.kernsize_conv2, fc1out_fc2in=run.fc1out_fc2in,
    #                   fc2out_outin=run.fc2out_outin, out_feat=run.out_feat)
    network = Network()
    print('network:', type(network))
    loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size)  # , shuffle=run.shuffle, drop_last=True)
    print('loader type:', type(loader))
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(3):
        m.begin_epoch()
        for batch in loader:

            images = batch[0]
            labels = batch[1]
            # images_on_gpu = images.cuda()
            # labels_on_gpu = labels.cuda()
            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # calc. Loss
            optimizer.zero_grad()
            loss.backward()  # calc. gradients
            optimizer.step()  # update weights

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
# m.save('results/results')

print("time elapsed: {:.2f}s".format(time.time() - start_time))

# PATH = './CurrentSavedModel.pth'  # give it a certain name, and saving it on another place? how to access each layer??
# torch.save(network.state_dict(), PATH)

# more accurate:
# torch.save({'epoch': epoch,   # if you want to create a checkpoint for continuing training
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, PATH)


# tensorboard --logdir=runs
