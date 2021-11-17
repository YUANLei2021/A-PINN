import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch.nn as fun
import os


def main():
    #  pde parameters
    lam = [1.0, 0.1]
    x_range = [0.0, 5.0]
    pde_parameters = [lam, x_range]

    # train parameters
    nf = 50
    optimizer = 0  # 0: L-BFGS 1: Adam 2: SGD
    max_iter = 500
    min_loss = 1e-8
    learning_rate = 0.01
    process = True
    train_parameters = [nf, optimizer, max_iter, min_loss, learning_rate, process]

    # test parameters
    dx = 0.01
    test_parameters = [dx]

    # Neural networks parameters
    nn_layers = [1, 40, 40, 2]  # neural networks layers
    act_fun = 0  # 0: fun.Tanh(), 1: fun.Sigmoid(), 2: fun.ReLU(), 3: Sin()
    nn_parameters = [nn_layers, act_fun]

    dirs = ['./model', './data', './png']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # train(pde_parameters, train_parameters, nn_parameters)  # train nn model
    test(pde_parameters, test_parameters)


def initial_u(x_i):
    u_i = torch.ones_like(x_i)
    return u_i


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


def pred(net_model, x, x_range):
    x_lb = x_range[0]
    x_ub = x_range[1]
    x_ = 2.0 * (x - x_lb) / (x_ub - x_lb) - 1.0
    model_return = net_model(x_)
    u, v = torch.split(model_return, 1, dim=1)
    return u, v


def train(pde_parameters, train_parameters, nn_parameters):
    # loading parameters
    [nf, opt, max_iter, min_loss, learning_rate, process] = train_parameters
    [lam, x_range] = pde_parameters
    # choose device to train model
    device = 'cpu'
    # x t input
    x_all = np.linspace(x_range[0], x_range[1], 10001).reshape(-1, 1)
    idx = np.random.choice(np.shape(x_all)[0], nf)
    x_star = x_all[idx, :]
    u_star = np.exp(-x_star) * np.cosh(x_star)
    noise = 0.00
    normal_data = np.random.normal(0, 1, u_star.shape)
    u_star = u_star * (1 + noise * normal_data)

    x_i = torch.FloatTensor(np.ones([1, 1]).reshape(-1, 1) * x_range[0]).to(device)
    x_f = torch.FloatTensor(x_star).to(device)
    u_f = torch.FloatTensor(u_star).to(device)
    training_points = np.append(x_star, u_star, axis=1)
    # dynamic display during training
    x_test = np.linspace(x_range[0], x_range[1], 100)
    x_test_ts = torch.FloatTensor(x_test.flatten()[:, None]).to(device)

    # set x t require grad
    x_i.requires_grad = True
    x_f.requires_grad = True
    # initialize neural networks and optimizer
    net_model = Net(nn_parameters).to(device)
    # add hyper-parameters to NN
    lam_f = torch.nn.Parameter(torch.FloatTensor(lam), requires_grad=True)
    net_model.register_parameter('lam', lam_f)
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
    lam_record = np.empty([0, 2])
    plt.ion()
    print('------------------------Neural networks------------------------------------')
    print(net_model)
    print('----------------------------Optimizer--------------------------------------')
    print(optimizer)
    print('------------------------Training device: ', device, '-----------------------------')
    #  -----------   start training   ------------
    starttime_train = datetime.datetime.now()
    print('------------------------Start training:{}---------------------'.format(starttime_train))

    while it < max_iter and loss > min_loss:
        def closure():
            u_i_pred, v_i_pred = pred(net_model, x_i, x_range)
            u_f_pred, v_f_pred = pred(net_model, x_f, x_range)

            v_i_pred_dx = torch.autograd.grad(v_i_pred.sum(), x_i, create_graph=True)[0]
            u_f_pred_dx = torch.autograd.grad(u_f_pred.sum(), x_f, create_graph=True)[0]
            v_f_pred_dx = torch.autograd.grad(v_f_pred.sum(), x_f, create_graph=True)[0]

            f = u_f_pred_dx + lam[0] * u_f_pred - lam_f[1] * v_f_pred
            e_outputs = v_f_pred_dx - u_f_pred + v_f_pred
            e_f = u_f_pred - u_f
            e_outputs2 = v_i_pred_dx - u_i_pred * torch.exp(x_i)
            loss_i = loss_func(v_i_pred, torch.zeros_like(x_i))
            loss_u_f = loss_func(e_f, torch.zeros_like(x_f))
            loss_f = loss_func(f, torch.zeros_like(x_f))
            loss_o = loss_func(e_outputs, torch.zeros_like(x_f)) + loss_func(e_outputs2, torch.zeros_like(x_i))

            loss_all = np.array((loss_u_f.data.numpy(), loss_f.data.numpy(), loss_o.data.numpy(), loss_i.data.numpy()))
            loss_min = np.min(loss_all)
            loss_all = loss_all / loss_min
            loss_all[loss_all > 10] = 10.0
            loss_weight = loss_all

            loss_total = 1.0 * loss_u_f + loss_weight[1] * loss_f + loss_weight[2] * loss_o + loss_weight[3] * loss_i
            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            return loss_total

        optimizer.step(closure)
        loss_value = closure().cpu().data.numpy()
        step_time = datetime.datetime.now() - starttime_train
        loss_record = np.append(loss_record, [[it, step_time.seconds + step_time.microseconds / 1000000, loss_value]],
                                axis=0)
        lam_record = np.append(lam_record, lam_f.data.numpy().reshape(1, 2), axis=0)
        if it % 10 == 0:
            print('Running: ', it, ' / ', max_iter)
            if process:
                plt.clf()
                u_test_pred, _ = pred(net_model, x_test_ts, x_range)
                u_test_pred = u_test_pred.data.numpy()
                plt.plot(x_test, u_test_pred, label='Pred value')
                plt.scatter(x_star, u_star, s=30, c='k', marker='x', label='Training points')
                plt.legend(loc='upper right')
                plt.pause(0.1)
        it = it + 1
    endtime_train = datetime.datetime.now()
    print('---------------End training:{}---------------'.format(endtime_train))
    torch.save(net_model, './model/A-PINN_D_VIDE_1D.pkl')
    np.savetxt('./data/A-PINN-D(noise)—VIDE-loss.txt', loss_record)
    np.savetxt('./data/A-PINN_D_VIDE-2D-lam.txt', lam_record)
    np.savetxt('./data/training points(noise).txt', training_points)
    train_time = endtime_train - starttime_train
    print('---------------Training time:{}s---------------'.format(train_time.seconds + train_time.microseconds / 1e6))

    plt.ioff()
    plt.show()

    plt.figure()
    plt.plot(loss_record[:, 0], np.log(loss_record[:, 2]))
    plt.savefig('./png/A-PINN_D_VIDE_1D-loss.png')
    plt.show()


def test(pde_parameters, test_parameters):
    [_, x_range] = pde_parameters
    [dx] = test_parameters
    net_model = torch.load('./model/A-PINN_D_VIDE_1D.pkl')
    device = 'cpu'
    x_test = np.linspace(x_range[0], x_range[1], int(1 / dx))
    x_test_ts = torch.FloatTensor(x_test.flatten()[:, None]).to(device)

    u_test_pred, _ = pred(net_model, x_test_ts, x_range)
    u_test_pred = u_test_pred.data.numpy().flatten()
    u_exact = np.exp(-x_test) * np.cosh(x_test)

    l2error = np.linalg.norm(u_test_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
    print('L2 error: ', l2error)
    plt.figure()
    plt.plot(x_test, u_test_pred, label='Pred value')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('./png/A-PINN_D_VIDE_1D-u(x).png')
    plt.show()


if __name__ == '__main__':
    main()
    pass
