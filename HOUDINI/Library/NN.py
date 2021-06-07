
import math
import json
import os
import numpy
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from HOUDINI.Library.Utils import NNUtils


class SaveableNNModule(nn.Module):
    def __init__(self, params_dict: dict = None):
        self.params_dict = params_dict
        super(SaveableNNModule, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def load(self, filename):
        self.load_state_dict(torch.load(
            filename, map_location=lambda storage, loc: storage))

    def save(self, directory):
        if directory[-1] == '/':
            directory = directory[:-1]

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = "{}/{}.pth".format(directory, self.name)
        torch.save(self.state_dict(), file_path)

        if self.params_dict is not None:
            if "output_activation" not in self.params_dict or self.params_dict["output_activation"] is None:
                self.params_dict["output_activation"] = "None"
            elif self.params_dict["output_activation"] == torch.sigmoid:
                self.params_dict["output_activation"] = "sigmoid"
            elif type(self.params_dict["output_activation"]) == nn.Softmax:
                self.params_dict["output_activation"] = "softmax"
            else:
                raise NotImplementedError

            jsond = json.dumps(self.params_dict)
            f = open("{}/{}.json".format(directory, self.name), "w")
            f.write(jsond)
            f.close()


# class NetCNN(SaveableNNModule):
#     def __init__(self, name, input_dim, input_ch):
#         """
#         :param output_activation: [None, F.softmax, torch.sigmoid]
#         """
#         super(NetCNN, self).__init__()

#         self.name = name
#         # self.layer_sizes = [64, 32]
#         self.layer_sizes = [32, 64]

#         self.conv1 = nn.Conv2d(input_ch, self.layer_sizes[0], kernel_size=5)
#         conv1_output_dim = self.cnn_get_output_dim(
#             input_dim, 5, stride=1, padding=0)
#         pool1_output_dim = self.cnn_get_output_dim(
#             conv1_output_dim, 2, stride=2, padding=0)

#         self.conv2 = nn.Conv2d(
#             self.layer_sizes[0], self.layer_sizes[1], kernel_size=5)
#         conv2_output_dim = self.cnn_get_output_dim(
#             pool1_output_dim, kernel_size=5, stride=1, padding=0)
#         self.pool2_output_dim = self.cnn_get_output_dim(
#             conv2_output_dim, 2, stride=2, padding=0)

#         self.conv2_drop = nn.Dropout2d()
#         """
#         self.fc1 = nn.Linear(self.pool2_output_dim ** 2 * self.layer_sizes[1], 1024)
#         self.bn1 = nn.BatchNorm1d(self.layer_sizes[2])
#         if self.output_dim is not None:
#             self.fc2 = nn.Linear(1024, output_dim)
#         """

#     def cnn_get_output_dim(self, w1, kernel_size, stride, padding=0):
#         w2 = (w1 - kernel_size + 2*padding) // stride + 1
#         return w2

#     def forward(self, x):
#         if type(x) == tuple:
#             x = x[1]

#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         return x


class NetRNN(SaveableNNModule):
    def __init__(self, name, input_dim, hidden_dim, output_dim=None, output_activation=None, output_sequence=False):
        """
        A function which goes from a list of items to a single hidden representation
        """
        super(NetRNN, self).__init__()
        self.name = name
        self.hidden_dim = hidden_dim

        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim)
        self.hidden = None  # a placeholder, used for the hidden state of the lstm
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.output_sequence = output_sequence
        if self.output_dim is not None:  # then we need an mlp at the end
            self.mlp = NetMLP(name="{}_mlp".format(self.name), input_dim=hidden_dim, output_dim=self.output_dim,
                              output_activation=self.output_activation, hidden_layer=False)

    def reset_hidden_state(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        t1 = torch.zeros(1, batch_size, self.hidden_dim)
        t2 = torch.zeros(1, batch_size, self.hidden_dim)

        if torch.cuda.is_available():
            #print("converting to cuda")
            t1.cuda()
            t2.cuda()

        var1 = Variable(t1)
        var2 = Variable(t2)
        self.hidden = (var1.cuda(), var2.cuda()
                       ) if torch.cuda.is_available() else (var1, var2)

    def forward(self, x):
        # if necessarry, concatenate from list to tensor
        if type(x) == list:
            if x.__len__() > 0 and type(x[0]) == tuple:
                x = [i[1] for i in x]
            x = [torch.unsqueeze(a, dim=1) for a in x]
            x = torch.cat(x, dim=1)

        # find out the batch_size and use it to reset the hidden state
        batch_size = x.data.shape[0]
        self.reset_hidden_state(batch_size)

        # transpose to a suitable form [list_size, batsh_size, items]
        x = torch.transpose(x, 0, 1)

        # apply rnn list_size number of times
        # lstm_output: [seq_len, batch, hidden_size * num_directions]
        # lstm_output: [seq_len, batch_size, hidden_size]
        # print(x)
        # print(self.hidden)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        last_hidden_state = lstm_out[-1]

        if self.output_dim is None:
            if not self.output_sequence:
                return last_hidden_state
            else:
                return lstm_out.transpose(0, 1)

        if not self.output_sequence:
            return self.mlp(last_hidden_state)

        # at this point, we need to output a sequence, so we use the mlp to process the all sequences
        seq_len = lstm_out.shape[0]
        grid_size = int(math.sqrt(seq_len))
        batch_size = lstm_out.shape[1]

        outputs = lstm_out.view(seq_len*batch_size, self.hidden_dim)
        outputs = self.mlp(outputs)
        if type(outputs) == tuple:
            # might want to keep the tuples, so they can be processed as well. not now.
            outputs = outputs[1]
        outputs = outputs.view(seq_len, batch_size)
        outputs = outputs.transpose(0, 1)   # batch_size, length_size
        outputs = outputs.contiguous().view(batch_size, grid_size, grid_size)
        return outputs


class NetDO(SaveableNNModule):
    def __init__(self,
                 name,
                 input_dim,
                 dt_name):
        super(NetDO, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.dt_name = dt_name

        self.cau_wei = nn.Parameter(data=torch.zeros(
            1, input_dim), requires_grad=True)
        self.cau_msk = nn.Parameter(data=torch.ones(
            1, input_dim), requires_grad=False)

    def forward(self, x):
        if type(x) == tuple:
            x = x[0]

        xprob = torch.sigmoid(self.cau_wei)
        out = self.cau_msk * xprob * x

        return out, xprob.detach().cpu().numpy()


class NetMLP(SaveableNNModule):
    def __init__(self,
                 name,
                 input_dim,
                 output_dim,
                 dt_name):
        super(NetMLP, self).__init__()
        self.name = name
        self.dt_name = dt_name
        self.fc0 = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        if type(x) == tuple:
            x, x_prob = x
        else:
            x_prob = None
        
        if self.dt_name == 'portec':
            x1 = self.fc(self.relu(self.fc0(x)))
        elif self.dt_name == 'lganm':
            x1 = self.fc(x)
        else:
            raise NotImplementedError()

        return x1, x_prob


class NetCNN(SaveableNNModule):
    def __init__(self, 
                 name, 
                 input_dim,
                 output_dim):
        super(NetCNN, self).__init__()
        mid_dim = (input_dim + output_dim) // 2

        kern0 = self.get_kern_size(input_dim, mid_dim)
        self.conv0 = nn.Conv1d(1, 8, kern0)

        kern1 = self.get_kern_size(mid_dim, output_dim)
        self.conv1 = nn.Conv1d(8, 1, kern1)

        self.relu = nn.ReLU()

    def get_kern_size(self, 
                      input_dim, 
                      output_dim, 
                      stride=1, 
                      padding=0):
        # easier to compute kernel size 
        assert stride == 1
        kern_size = input_dim - output_dim + 2 * padding + 1
        return kern_size

    def forward(self, x):
        if type(x) == tuple:
            x, x_prob = x
        else:
            x_prob = None

        x = torch.unsqueeze(x, dim=1)
        x1 = self.conv1(self.relu(self.conv0(x)))
        x1 = torch.squeeze(x1, dim=1)
        return x1, x_prob
