import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as fun


def main():
    dx = 0.01
    x_range = [0.0, 5.0]
    test(dx, x_range)
    pass


class Net(torch.nn.Module):
    def __init__(self, parameters):
        [nn_layers, act_fun] = parameters
        af_list = {
            0: fun.Tanh(),
            1: fun.Sigmoid(),
            2: fun.ReLU(),
            # 3: Sin()
        }
        activation_function = af_list.get(act_fun, None)
        super(Net, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(nn_layers) - 2):
            self.layers.append(torch.nn.Linear(nn_layers[i], nn_layers[i + 1]), )
            self.layers.append(activation_function)
        self.layers.append(torch.nn.Linear(nn_layers[i + 1], nn_layers[i + 2]), )

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def pred(net_model, x, x_range):  # 先归一化 再预测值
    x_lb = x_range[0]
    x_ub = x_range[1]
    x_ = 2.0 * (x - x_lb) / (x_ub - x_lb) - 1.0
    model_return = net_model(x_)
    u, v = torch.split(model_return, 1, dim=1)
    return u, v


def test(dx, x_range):
    net_model = torch.load('./model/A-PINN_D_VIDE_1D.pkl')
    device = 'cpu'
    x_test = np.linspace(x_range[0], x_range[1], int(1 / dx))
    x_test_ts = torch.FloatTensor(x_test.flatten()[:, None]).to(device)  # 网格拍平

    u_pred_noise, _ = pred(net_model, x_test_ts, x_range)
    u_pred_noise = u_pred_noise.data.numpy().flatten()
    tp_noise = np.loadtxt('./data/training points(noise).txt')

    plt.figure(figsize=(8, 6))
    plt.plot(x_test, u_pred_noise, label='Predictive u(x) of A-PINN')
    plt.scatter(tp_noise[:, 0], tp_noise[:, 1], s=30, c='k', marker='x', label='Training data without noise')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.xlim(x_range[0], x_range[1])
    plt.tight_layout()
    plt.savefig('./png/A-PINN_D_VIDE_1D-u(x).png')
    plt.show()


if __name__ == '__main__':
    main()
    pass
