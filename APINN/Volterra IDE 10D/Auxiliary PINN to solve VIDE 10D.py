import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch.nn as fun
import os


def main():

    # train parameters
    nb = 100
    nf = 10000
    optimizer = 0  # 0: L-BFGS 1: Adam 2: SGD
    max_iter = 1000
    min_loss = 1e-8
    learning_rate = 0.01
    process = False
    train_parameters = [nb, nf, optimizer, max_iter, min_loss, learning_rate, process]

    # Neural networks parameters
    nn_layers = [10, 40, 40, 40, 40, 11]  # neural networks layers
    act_fun = 0  # 0: fun.Tanh(), 1: fun.Sigmoid(), 2: fun.ReLU(), 3: Sin()
    nn_parameters = [nn_layers, act_fun]

    dirs = ['./model', './data', './png']
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # train(train_parameters, nn_parameters)  # train nn model
    test()
    test2()


class Net(torch.nn.Module):
    def __init__(self, parameters):
        [nn_layers, act_fun] = parameters
        af_list = {
            0: fun.Tanh(),
            1: fun.Sigmoid(),
            2: fun.ReLU(),
            3: Sin()
        }
        activation_function = af_list.get(act_fun, None)
        super(Net, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(nn_layers) - 2):
            self.layers.append(torch.nn.Linear(nn_layers[i], nn_layers[i + 1]), )
            self.layers.append(activation_function)
        self.layers.append(torch.nn.Linear(nn_layers[-2], nn_layers[-1]), )

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class Sin(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.sin(x)


def pred(net_model, t, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    model_return = net_model(torch.cat((t, x1, x2, x3, x4, x5, x6, x7, x8, x9), 1))
    u, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = torch.split(model_return, 1, dim=1)
    return u, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10


def train(train_parameters, nn_parameters):
    # loading parameters
    [nb, nf, opt, max_iter, min_loss, learning_rate, process] = train_parameters
    # choose device to train model
    device = 'cpu'
    # x t input
    t_b1 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x1_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b1 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b2 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x2_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b2 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b3 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x3_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b3 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b4 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x4_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b4 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b5 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x5_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b5 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b6 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x6_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b6 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b7 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x7_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b7 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b8 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x8_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b8 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b9 = torch.FloatTensor(np.zeros([nb, 1])).to(device)
    x9_b9 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)

    t_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x1_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x2_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x3_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x4_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x5_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x6_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x7_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x8_b10 = torch.FloatTensor(np.random.random(nb).reshape(-1, 1)).to(device)
    x9_b10 = torch.FloatTensor(np.zeros([nb, 1])).to(device)

    t_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x1_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x2_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x3_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x4_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x5_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x6_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x7_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x8_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x9_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)

    u_exact = (t_f * (x1_f + x2_f + x3_f) * torch.sin(x4_f + x5_f + x6_f) * torch.cos(x7_f + x8_f + x9_f)).data.numpy()

    g_f = t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x4_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x7_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x4_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x8_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x4_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x9_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x7_f + x8_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x7_f + x9_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(
        x8_f + x9_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x5_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x5_f + x6_f) / 4 + t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f * x3_f ** 2 * torch.sin(x7_f + x8_f + x9_f) / 4 - 3 * t_f * (
                      x1_f + x2_f + x3_f) * torch.sin(x4_f + x5_f + x6_f) * torch.sin(x7_f + x8_f + x9_f) - t_f * (
                      x1_f + x2_f + x3_f) * torch.sin(x4_f + x5_f + x6_f) * torch.cos(x7_f + x8_f + x9_f) + 3 * t_f * (
                      x1_f + x2_f + x3_f) * torch.cos(x4_f + x5_f + x6_f) * torch.cos(
        x7_f + x8_f + x9_f) + 3 * t_f * torch.sin(x4_f + x5_f + x6_f) * torch.cos(x7_f + x8_f + x9_f) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(
                  x7_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(
                  x8_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(
                  x9_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(
        x7_f + x8_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(
                  x7_f + x9_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(
                  x8_f + x9_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(
                  x7_f + x8_f + x9_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
                  x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(x4_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(x5_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(
                  x4_f + x5_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(x4_f + x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(
                  x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(x4_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(x5_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(
                  x4_f + x5_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(x4_f + x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(
                  x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(x4_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(x5_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(
                  x4_f + x5_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(x4_f + x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(
                  x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(x4_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(x5_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
                  x4_f + x5_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(x4_f + x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
                  x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(x4_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(x5_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(x4_f + x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
                  x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(x4_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(x5_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
                  x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(x4_f + x6_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
                  x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
                  x5_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(x4_f) / 4) - x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
                  x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
                  x5_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(x6_f) / 4) + x3_f * (
                      -t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
                  x4_f + x5_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x6_f) / 4) + x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x5_f + x6_f) / 4) - x3_f * (-t_f ** 3 * x1_f ** 2 * x2_f * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4 - t_f ** 3 * x1_f * x2_f ** 2 * torch.sin(x7_f + x8_f + x9_f) * torch.cos(
        x4_f + x5_f + x6_f) / 4) + (x1_f + x2_f + x3_f) * torch.sin(x4_f + x5_f + x6_f) * torch.cos(x7_f + x8_f + x9_f)

    # set x t require grad
    t_f.requires_grad = True
    x1_f.requires_grad = True
    x2_f.requires_grad = True
    x3_f.requires_grad = True
    x4_f.requires_grad = True
    x5_f.requires_grad = True
    x6_f.requires_grad = True
    x7_f.requires_grad = True
    x8_f.requires_grad = True
    x9_f.requires_grad = True
    # initialize neural networks and optimizer
    net_model = Net(nn_parameters).to(device)
    # choose optimizer
    if opt == 1:
        optimizer = torch.optim.Adam([{'params': net_model.parameters()}], lr=learning_rate)
    elif opt == 2:
        optimizer = torch.optim.SGD([{'params': net_model.parameters()}], lr=learning_rate)
    else:
        optimizer = torch.optim.LBFGS([{'params': net_model.parameters()}], lr=learning_rate)
    # set loss function
    loss_func = torch.nn.MSELoss().to(device)

    it = 0
    loss = 10
    loss_record = np.empty([0, 3])
    print('------------------------Neural networks------------------------------------')
    print(net_model)
    print('----------------------------Optimizer--------------------------------------')
    print(optimizer)
    print('------------------------Training device: ', device, '-----------------------------')
    #  -----------   start training  ------------
    start_time = datetime.datetime.now()
    print('------------------------Start training:{}---------------------'.format(start_time))

    while it < max_iter and loss > min_loss:
        def closure():
            u_b1, _, _, _, _, _, _, _, _, _, v10_b1 = pred(net_model, t_b1, x1_b1, x2_b1, x3_b1, x4_b1, x5_b1, x6_b1,
                                                           x7_b1, x8_b1, x9_b1)
            u_b2, _, _, _, _, _, _, _, _, v9_b2, _ = pred(net_model, t_b2, x1_b2, x2_b2, x3_b2, x4_b2, x5_b2, x6_b2,
                                                          x7_b2, x8_b2, x9_b2)
            u_b3, _, _, _, _, _, _, _, v8_b3, _, _ = pred(net_model, t_b3, x1_b3, x2_b3, x3_b3, x4_b3, x5_b3, x6_b3,
                                                          x7_b3, x8_b3, x9_b3)
            u_b4, _, _, _, _, _, _, v7_b4, _, _, _ = pred(net_model, t_b4, x1_b4, x2_b4, x3_b4, x4_b4, x5_b4, x6_b4,
                                                          x7_b4, x8_b4, x9_b4)
            u_b5, _, _, _, _, _, v6_b5, _, _, _, _ = pred(net_model, t_b5, x1_b5, x2_b5, x3_b5, x4_b5, x5_b5, x6_b5,
                                                          x7_b5, x8_b5, x9_b5)
            u_b6, _, _, _, _, v5_b6, _, _, _, _, _ = pred(net_model, t_b6, x1_b6, x2_b6, x3_b6, x4_b6, x5_b6, x6_b6,
                                                          x7_b6, x8_b6, x9_b6)
            u_b7, _, _, _, v4_b7, _, _, _, _, _, _ = pred(net_model, t_b7, x1_b7, x2_b7, x3_b7, x4_b7, x5_b7, x6_b7,
                                                          x7_b7, x8_b7, x9_b7)
            u_b8, _, _, v3_b8, _, _, _, _, _, _, _ = pred(net_model, t_b8, x1_b8, x2_b8, x3_b8, x4_b8, x5_b8, x6_b8,
                                                          x7_b8, x8_b8, x9_b8)
            u_b9, _, v2_b9, _, _, _, _, _, _, _, _ = pred(net_model, t_b9, x1_b9, x2_b9, x3_b9, x4_b9, x5_b9, x6_b9,
                                                          x7_b9, x8_b9, x9_b9)
            u_b10, v1_b10, _, _, _, _, _, _, _, _, _ = pred(net_model, t_b10, x1_b10, x2_b10, x3_b10, x4_b10, x5_b10,
                                                            x6_b10, x7_b10, x8_b10, x9_b10)

            u_f, v1_f, v2_f, v3_f, v4_f, v5_f, v6_f, v7_f, v8_f, v9_f, v10_f = pred(net_model, t_f, x1_f, x2_f, x3_f,
                                                                                    x4_f, x5_f, x6_f, x7_f, x8_f, x9_f)

            u_f_dt = torch.autograd.grad(u_f.sum(), t_f, create_graph=True)[0]
            u_f_dx1 = torch.autograd.grad(u_f.sum(), x1_f, create_graph=True)[0]
            u_f_dx2 = torch.autograd.grad(u_f.sum(), x2_f, create_graph=True)[0]
            u_f_dx3 = torch.autograd.grad(u_f.sum(), x3_f, create_graph=True)[0]
            u_f_dx4 = torch.autograd.grad(u_f.sum(), x4_f, create_graph=True)[0]
            u_f_dx5 = torch.autograd.grad(u_f.sum(), x5_f, create_graph=True)[0]
            u_f_dx6 = torch.autograd.grad(u_f.sum(), x6_f, create_graph=True)[0]
            u_f_dx7 = torch.autograd.grad(u_f.sum(), x7_f, create_graph=True)[0]
            u_f_dx8 = torch.autograd.grad(u_f.sum(), x8_f, create_graph=True)[0]
            u_f_dx9 = torch.autograd.grad(u_f.sum(), x9_f, create_graph=True)[0]

            v1_f_dx9 = torch.autograd.grad(v1_f.sum(), x9_f, create_graph=True)[0]
            v2_f_dx8 = torch.autograd.grad(v2_f.sum(), x8_f, create_graph=True)[0]
            v3_f_dx7 = torch.autograd.grad(v3_f.sum(), x7_f, create_graph=True)[0]
            v4_f_dx6 = torch.autograd.grad(v4_f.sum(), x6_f, create_graph=True)[0]
            v5_f_dx5 = torch.autograd.grad(v5_f.sum(), x5_f, create_graph=True)[0]
            v6_f_dx4 = torch.autograd.grad(v6_f.sum(), x4_f, create_graph=True)[0]
            v7_f_dx3 = torch.autograd.grad(v7_f.sum(), x3_f, create_graph=True)[0]
            v8_f_dx2 = torch.autograd.grad(v8_f.sum(), x2_f, create_graph=True)[0]
            v9_f_dx1 = torch.autograd.grad(v9_f.sum(), x1_f, create_graph=True)[0]
            v10_f_dt = torch.autograd.grad(v10_f.sum(), t_f, create_graph=True)[0]

            f = u_f_dt + u_f_dx1 + u_f_dx2 + u_f_dx3 + u_f_dx4 + u_f_dx5 + u_f_dx6 + u_f_dx7 + u_f_dx8 + u_f_dx9 - u_f - g_f - v1_f
            e_b1 = u_b1 - t_b1 * (x1_b1 + x2_b1 + x3_b1) * torch.sin(x4_b1 + x5_b1 + x6_b1) * torch.cos(
                x7_b1 + x8_b1 + x9_b1)
            e_b2 = u_b2 - t_b2 * (x1_b2 + x2_b2 + x3_b2) * torch.sin(x4_b2 + x5_b2 + x6_b2) * torch.cos(
                x7_b2 + x8_b2 + x9_b2)
            e_b3 = u_b3 - t_b3 * (x1_b3 + x2_b3 + x3_b3) * torch.sin(x4_b3 + x5_b3 + x6_b3) * torch.cos(
                x7_b3 + x8_b3 + x9_b3)
            e_b4 = u_b4 - t_b4 * (x1_b4 + x2_b4 + x3_b4) * torch.sin(x4_b4 + x5_b4 + x6_b4) * torch.cos(
                x7_b4 + x8_b4 + x9_b4)
            e_b5 = u_b5 - t_b5 * (x1_b5 + x2_b5 + x3_b5) * torch.sin(x4_b5 + x5_b5 + x6_b5) * torch.cos(
                x7_b5 + x8_b5 + x9_b5)
            e_b6 = u_b6 - t_b6 * (x1_b6 + x2_b6 + x3_b6) * torch.sin(x4_b6 + x5_b6 + x6_b6) * torch.cos(
                x7_b6 + x8_b6 + x9_b6)
            e_b7 = u_b7 - t_b7 * (x1_b7 + x2_b7 + x3_b7) * torch.sin(x4_b7 + x5_b7 + x6_b7) * torch.cos(
                x7_b7 + x8_b7 + x9_b7)
            e_b8 = u_b8 - t_b8 * (x1_b8 + x2_b8 + x3_b8) * torch.sin(x4_b8 + x5_b8 + x6_b8) * torch.cos(
                x7_b8 + x8_b8 + x9_b8)
            e_b9 = u_b9 - t_b9 * (x1_b9 + x2_b9 + x3_b9) * torch.sin(x4_b9 + x5_b9 + x6_b9) * torch.cos(
                x7_b9 + x8_b9 + x9_b9)
            e_b10 = u_b10 - t_b10 * (x1_b10 + x2_b10 + x3_b10) * torch.sin(x4_b10 + x5_b10 + x6_b10) * torch.cos(
                x7_b10 + x8_b10 + x9_b10)

            e_b11 = v1_b10 - 0
            e_b12 = v2_b9 - 0
            e_b13 = v3_b8 - 0
            e_b14 = v4_b7 - 0
            e_b15 = v5_b6 - 0
            e_b16 = v6_b5 - 0
            e_b17 = v7_b4 - 0
            e_b18 = v8_b3 - 0
            e_b19 = v9_b2 - 0
            e_b20 = v10_b1 - 0

            e_outputs1 = v1_f_dx9 - v2_f
            e_outputs2 = v2_f_dx8 - v3_f
            e_outputs3 = v3_f_dx7 - v4_f
            e_outputs4 = v4_f_dx6 - v5_f
            e_outputs5 = v5_f_dx5 - v6_f
            e_outputs6 = v6_f_dx4 - v7_f
            e_outputs7 = v7_f_dx3 - v8_f
            e_outputs8 = v8_f_dx2 - v9_f
            e_outputs9 = v9_f_dx1 - v10_f
            e_outputs10 = v10_f_dt - u_f

            loss_b1 = loss_func(e_b1, torch.zeros_like(x1_b1)) + loss_func(e_b2, torch.zeros_like(x1_b2)) + \
                      loss_func(e_b3, torch.zeros_like(x1_b3)) + loss_func(e_b4, torch.zeros_like(x1_b3)) + \
                      loss_func(e_b5, torch.zeros_like(x1_b2)) + loss_func(e_b6, torch.zeros_like(x1_b2)) + \
                      loss_func(e_b7, torch.zeros_like(x1_b2)) + loss_func(e_b8, torch.zeros_like(x1_b2)) + \
                      loss_func(e_b9, torch.zeros_like(x1_b2)) + loss_func(e_b10, torch.zeros_like(x1_b2))
            loss_b2 = loss_func(e_b11, torch.zeros_like(x1_b4)) + loss_func(e_b12, torch.zeros_like(x1_b3)) + \
                      loss_func(e_b13, torch.zeros_like(x1_b2)) + loss_func(e_b14, torch.zeros_like(x1_b1)) + \
                      loss_func(e_b15, torch.zeros_like(x1_b2)) + loss_func(e_b16, torch.zeros_like(x1_b1)) + \
                      loss_func(e_b17, torch.zeros_like(x1_b2)) + loss_func(e_b18, torch.zeros_like(x1_b1)) + \
                      loss_func(e_b19, torch.zeros_like(x1_b2)) + loss_func(e_b20, torch.zeros_like(x1_b1))
            loss_b = loss_b1 + loss_b2
            loss_f = loss_func(f, torch.zeros_like(x1_f))

            loss_o = loss_func(e_outputs1, torch.zeros_like(x1_f)) + loss_func(e_outputs2, torch.zeros_like(x1_f)) + \
                     loss_func(e_outputs3, torch.zeros_like(x1_f)) + loss_func(e_outputs4, torch.zeros_like(x1_f)) + \
                     loss_func(e_outputs5, torch.zeros_like(x1_f)) + loss_func(e_outputs6, torch.zeros_like(x1_f)) + \
                     loss_func(e_outputs7, torch.zeros_like(x1_f)) + loss_func(e_outputs8, torch.zeros_like(x1_f)) + \
                     loss_func(e_outputs9, torch.zeros_like(x1_f)) + loss_func(e_outputs10, torch.zeros_like(x1_f))

            loss_all = np.array((loss_b.data.numpy(), loss_f.data.numpy(), loss_o.data.numpy()))
            loss_min = np.min(loss_all)
            loss_all = loss_all / loss_min
            loss_all[loss_all > 100] = 100.0
            loss_weight = loss_all

            loss_total = loss_weight[0] * loss_b + loss_weight[1] * loss_f + loss_weight[2] * loss_o
            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            return loss_total

        optimizer.step(closure)
        loss_value = closure().cpu().data.numpy()
        step_time = datetime.datetime.now() - start_time
        loss_record = np.append(loss_record, [[it, step_time.seconds + step_time.microseconds / 1000000, loss_value]],
                                axis=0)

        if it % 10 == 0:
            print('Running: ', it, ' / ', max_iter)
            if process:
                u_test_pred, _, _, _, _, _, _, _, _, _, _ = pred(net_model, t_f, x1_f, x2_f, x3_f, x4_f, x5_f, x6_f,
                                                                 x7_f, x8_f, x9_f)
                u_test_pred = u_test_pred.data.numpy()
                l2error = np.linalg.norm(u_test_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
                print('L2 error:', l2error)
        it = it + 1
    end_time = datetime.datetime.now()
    print('---------------End training:{}---------------'.format(end_time))
    torch.save(net_model, './model/A-PINN_S_10D-IDE.pkl')
    np.savetxt('./data/A-PINN_S_10D-VIDE-loss.txt', loss_record)

    train_time = end_time - start_time
    print('---------------Training time:{}s---------------'.format(train_time.seconds + train_time.microseconds / 1e6))


def test():
    x_range = (0, 1)
    t_range = (0, 1)
    x4_exact = np.arange(0, 1 + 0.01, 0.01)
    x5_exact = np.arange(0, 1 + 0.01, 0.01)
    x4_test = x4_exact.flatten()
    x5_test = x5_exact.flatten()

    net_model = torch.load('./model/A-PINN_S_10D-IDE.pkl')
    device = 'cpu'

    # first plane
    x4_mesh, x5_mesh = np.meshgrid(x4_test, x5_test)
    x4x5_shape = np.shape(x4_mesh)
    x4_test_ts = torch.FloatTensor(x4_mesh.flatten()[:, None]).to(device)
    x5_test_ts = torch.FloatTensor(x5_mesh.flatten()[:, None]).to(device)
    t_test_ts = torch.ones_like(x4_test_ts).to(device)
    x1_test_ts = torch.ones_like(x4_test_ts).to(device)
    x2_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x3_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x6_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x7_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x8_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x9_test_ts = torch.zeros_like(x4_test_ts).to(device)

    u_test = 1.0 * torch.sin(x4_test_ts + x5_test_ts)
    u_exact = u_test.data.numpy().reshape(x4x5_shape).T

    u_test_pred, _, _, _, _, _, _, _, _, _, _ = pred(net_model, t_test_ts, x1_test_ts, x2_test_ts, x3_test_ts,
                                                     x4_test_ts, x5_test_ts, x6_test_ts, x7_test_ts, x8_test_ts,
                                                     x9_test_ts)
    u_test_pred = u_test_pred.data.numpy().reshape(x4x5_shape).T

    abs_e1 = u_test_pred - u_exact
    u_test1 = u_test_pred

    # second plane
    t_test_ts = torch.ones_like(x4_test_ts).to(device)
    x1_test_ts = torch.ones_like(x4_test_ts).to(device)
    x2_test_ts = torch.ones_like(x4_test_ts).to(device)
    x3_test_ts = torch.ones_like(x4_test_ts).to(device)
    x6_test_ts = torch.ones_like(x4_test_ts).to(device)
    x7_test_ts = torch.ones_like(x4_test_ts).to(device)
    x8_test_ts = x4_test_ts
    x9_test_ts = x5_test_ts
    x4_test_ts = torch.zeros_like(x4_test_ts).to(device)
    x5_test_ts = torch.ones_like(x4_test_ts).to(device)

    u_test = 3.0 * np.sin(2.0) * torch.cos(x8_test_ts + x9_test_ts + 1.0)
    u_exact = u_test.data.numpy().reshape(x4x5_shape).T

    u_test_pred, _, _, _, _, _, _, _, _, _, _ = pred(net_model, t_test_ts, x1_test_ts, x2_test_ts, x3_test_ts,
                                                     x4_test_ts, x5_test_ts, x6_test_ts, x7_test_ts, x8_test_ts,
                                                     x9_test_ts)
    u_test_pred = u_test_pred.data.numpy().reshape(x4x5_shape).T

    abs_e2 = u_test_pred - u_exact
    u_test2 = u_test_pred

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    pic = plt.imshow(u_test1.T, cmap='jet', vmin=0.0, vmax=1.0,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('$ x_4 $')
    plt.ylabel('$ x_5 $')
    plt.title('u_pred', fontsize='small')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    pic = plt.imshow(abs_e1.T, cmap='jet', vmin=-0.02, vmax=0.02,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('$ x_4 $')
    plt.ylabel('$ x_5 $')
    plt.title('Error in u_pred', fontsize='small')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    pic = plt.imshow(u_test2.T, cmap='jet', vmin=-3.0, vmax=3.0,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('$ x_8 $')
    plt.ylabel('$ x_9 $')
    plt.title('u_pred', fontsize='small')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    pic = plt.imshow(abs_e2.T, cmap='jet', vmin=-0.03, vmax=0.03,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('$ x_8 $')
    plt.ylabel('$ x_9 $')
    plt.title('Error in u_pred', fontsize='small')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    plt.savefig('./png/A-PINN_S_10D-VIDE-u(x,t).png')
    plt.show()


def test2():
    net_model = torch.load('./model/A-PINN_S_10D-IDE.pkl')
    device = 'cpu'
    nf = 100000
    t_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x1_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x2_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x3_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x4_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x5_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x6_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x7_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x8_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)
    x9_f = torch.FloatTensor(np.random.random(nf).reshape(-1, 1)).to(device)

    u_exact = (t_f * (x1_f + x2_f + x3_f) * torch.sin(x4_f + x5_f + x6_f) * torch.cos(x7_f + x8_f + x9_f)).data.numpy()

    u_test_pred, v1_f, v2_f, v3_f, v4_f, v5_f, v6_f, v7_f, v8_f, v9_f, v10_f = pred(net_model, t_f, x1_f, x2_f, x3_f,
                                                                                    x4_f, x5_f, x6_f, x7_f, x8_f, x9_f)
    u_test_pred = u_test_pred.data.numpy()
    l2error = np.linalg.norm(u_test_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
    print('L2 error: ', l2error)


if __name__ == '__main__':
    main()
    pass
