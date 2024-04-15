import numpy as np
import torch
import torch.nn as nn


class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)

        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, norm=True):
        if norm:
            x = self.dropout(self.bn(self.act(self.lin(x))))
        else:
            x = self.act(self.lin(x))
        return x


class LinLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.bn(self.act(self.lin(x))))
        return x


class InputProcess(nn.Module):
    def __init__(self, **kwargs):
        super(InputProcess, self).__init__(**kwargs)

		# (n_var, eps, momentum)
        self.jet_bn = torch.nn.BatchNorm1d(6, eps=0.001, momentum=0.6)
        self.jet_conv1 = InputConv(6, 32)
        self.jet_conv2 = InputConv(32, 16)
        self.jet_conv3 = InputConv(16, 4)

        self.lepton_bn = torch.nn.BatchNorm1d(4, eps=0.001, momentum=0.6)
        self.lepton_conv1 = InputConv(4, 32)
        self.lepton_conv2 = InputConv(32, 16)
        self.lepton_conv3 = InputConv(16, 4)

    def forward(self, jet, lepton):

        jet = self.jet_bn(torch.transpose(jet, 1, 2))
        jet = self.jet_conv1(jet)
        jet = self.jet_conv2(jet)
        jet = self.jet_conv3(jet, norm=False)
        jet = torch.transpose(jet, 1, 2)

        lepton = self.lepton_bn(torch.transpose(lepton, 1, 2))
        lepton = self.lepton_conv1(lepton)
        lepton = self.lepton_conv2(lepton)
        lepton = self.lepton_conv3(lepton, norm=False)
        lepton = torch.transpose(lepton, 1, 2)

        #return cpf, npf, vtx
        return jet, lepton


class DenseClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)

        self.LinLayer1 = LinLayer(103, 50)
        self.LinLayer2 = LinLayer(50, 25)
        self.LinLayer3 = LinLayer(25, 25)
        self.LinLayer4 = LinLayer(25, 25)
        self.LinLayer5 = LinLayer(25, 25)
        self.LinLayer6 = LinLayer(25, 25)
        self.LinLayer7 = LinLayer(25, 25)
        self.LinLayer8 = LinLayer(25, 25)

    def forward(self, x):
        x = self.LinLayer1(x)
        x = self.LinLayer2(x)
        x = self.LinLayer3(x)
        x = self.LinLayer4(x)
        x = self.LinLayer5(x)
        x = self.LinLayer6(x)
        x = self.LinLayer7(x)
        x = self.LinLayer8(x)

        return x
