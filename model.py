""" Convolutional neural network. """
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


# try to pull out the variables from the network and create lists to test different network structures !

#
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12 * 4 * 4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

# if I import the Network from the training folder... a training cycle is started !
# But I just want to import the network


# pulling the variables out of the class


# class Network(nn.Module):
#     def __init__(self, input_chan, conv1out_conv2in, kernsize_conv1,
#                  conv2out_fc1in, kernsize_conv2, fc1out_fc2in,
#                  fc2out_outin, out_feat):
#         super(Network, self).__init__()
#         self.input_chan = input_chan
#         self.conv1out_conv2in = conv1out_conv2in
#         self.kernsize_conv1 = kernsize_conv1
#         self.conv2out_fc1in = conv2out_fc1in
#         self.kernsize_conv2 = kernsize_conv2
#         self.conv1 = nn.Conv2d(in_channels=input_chan, out_channels=conv1out_conv2in, kernel_size=kernsize_conv1)
#         self.conv2 = nn.Conv2d(in_channels=conv1out_conv2in, out_channels=conv2out_fc1in, kernel_size=kernsize_conv2)

#         self.fc1out_fc2in = fc1out_fc2in
#         self.fc2out_outin = fc2out_outin
#         self.out_feat = out_feat
#         self.fc1 = nn.Linear(in_features=conv2out_fc1in*(kernsize_conv1-1)*(kernsize_conv2-1), out_features=fc1out_fc2in)
#         self.fc2 = nn.Linear(in_features=fc1out_fc2in, out_features=fc2out_outin)
#         self.out = nn.Linear(in_features=fc2out_outin, out_features=out_feat)

#     def forward(self, t):
#         t = F.relu(self.conv1(t))
#         t = F.max_pool2d(t, kernel_size=2, stride=2)

#         t = F.relu(self.conv2(t))
#         t = F.max_pool2d(t, kernel_size=2, stride=2)

#         t = F.relu(self.fc1(t.reshape(-1, 12 * 4 * 4)))
#         t = F.relu(self.fc2(t))
#         t = self.out(t)
#         return t
#
#
# network = Network()

# pretty complex formulas ! not what I did there

# network = Network(input_chan=1,  # default values
#                   conv1out_conv2in=6,
#                   kernsize_conv1=5,
#                   conv2out_fc1in=12,
#                   kernsize_conv2=5,
#                   fc1out_fc2in=120,
#                   fc2out_outin=60,
#                   out_feat=10)
#
# network2 = Network(input_chan=1,   # this
#                    conv1out_conv2in=6,
#                    kernsize_conv1=5,
#                    conv2out_fc1in=12,
#                    kernsize_conv2=5,
#                    fc1out_fc2in=120,
#                    fc2out_outin=60,
#                    out_feat=10)
#
#
# Network_size_parameter_list = [network, network2]

# a list with predefined networks:
# easily adding new layers and define their size. these data are saved in the results file and in the name
# for tensorboard?!

# Network_list = []
# for network in Network_list:
#     Network_list.append()


# print(network.conv1.weight.shape)
# print(network.conv2.weight.shape)
# print(network.fc1.weight.shape)
# print(network.fc2.weight.shape)
# print(network.out.weight.shape)
# # print(network.conv1.weight)
# # print(network.conv2.weight)


# Network parameters to be iterated: (using list as datatype)
# input_chan       = [1]
# conv1out_conv2in = [6, 19]
# kernsize_conv1   = [5]
# conv2out_fc1in   = [12]
# kernsize_conv2   = [5]
# fc1out_fc2in     = [120]
# fc2out_outin     = [60]
# out_feat         = [10]
# # print(combinations)
# # list1 = list(combinations)
# # print(list1)
# # instead of a few lists, make a dictionary ! OrderedDict()
#
# import itertools
# for parameter_combi in itertools.product(input_chan, conv1out_conv2in, kernsize_conv1,
#                  conv2out_fc1in, kernsize_conv2, fc1out_fc2in,
#                  fc2out_outin, out_feat):
#     # print(parameter_combi)  # (1, 6, 5, 12, 5, 120, 60, 10)
#     network = Network(*parameter_combi)
#     print(network)  # works !!!
#     # -> 2 diff networks are created here. this for-loop can be used to search for different network-size-parameters

# saving the networks parameters as well. put a shortcut for its structure into the filnename/header whatever to keep
# everything clean
# !!!: the network is inaccurate: formulas have to be inserted into the network

# ------------
# next task: adding additional layers and turning them easily on/off, and or iterate over different layer numbers
# putting all parameters into the training file

# think about the logic of all parameters and improve them, learn from them


#
# input_chan       = [1]
# conv1out_conv2in = [6, 19]
# kernsize_conv1   = [5]
# conv2out_fc1in   = [12]
# kernsize_conv2   = [5]
# fc1out_fc2in     = [120]
# fc2out_outin     = [60]
# out_feat         = [10]
#
# from collections import OrderedDict
# params2 = OrderedDict(
#     input_chan=[1],
#     conv1out_conv2in=[6, 19],
#     kernsize_conv1=[5],
#     conv2out_fc1in=[12],
#     kernsize_conv2=[5],
#     fc1out_fc2in=[120],
#     fc2out_outin=[60],
#     out_feat=[10]
#
# )


# ok: now use this dict for iterating two networks
# runbuilder has to be modified to take only 1 network as well
# class RunBuilder():
#     @staticmethod
#     def get_runs(params):
#
#         Run = namedtuple('Run', params.keys())
#
#         runs = []
#         for v in product(*params.values()):
#             runs.append(Run(*v))
#
#         return runs


# loss function and optimizer have to be inserted manually?
# better to move these variables also out !


#
# from collections import OrderedDict
#
# params = OrderedDict(
#     lr=[.01],
#     batch_size=[1000],
#     shuffle=[False],
#     input_chan=[1],
#     conv1out_conv2in=[6, 19],
#     kernsize_conv1=[5],
#     conv2out_fc1in=[12],
#     kernsize_conv2=[5],
#     fc1out_fc2in=[120],
#     fc2out_outin=[60],
#     out_feat=[10]
# )
