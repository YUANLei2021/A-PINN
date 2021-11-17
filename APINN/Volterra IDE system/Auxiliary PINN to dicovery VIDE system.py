import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch.nn as fun
import os


def main():
    #  pde parameters
    lam = [1.0, 1.0]
    x_range = [0.0, 1.0]
    pde_parameters = [lam, x_range]

    # train parameters
    nf = 50
    optimizer = 0  # 0: L-BFGS 1: Adam 2: SGD
    max_iter = 500
    min_loss = 1e-8
    learning_rate = 0.1
    process = True
    train_parameters = [nf, optimizer, max_iter, min_loss, learning_rate, process]

    # test parameters
    dx = 0.05
    test_parameters = [dx]

    # Neural networks parameters
    nn_layers = [1, 40, 40, 4]  # neural networks layers
    act_fun = 0  # 0: fun.Tanh(), 1: fun.Sigmoid(), 2: fun.ReLU(), 3: Sin()
    nn_parameters = [nn_layers, act_fun]

    dirs = ['./model', './data', './png']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # train(pde_parameters, train_parameters, nn_parameters)  # train nn model
    test(pde_parameters, test_parameters)


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
    u, v, w, p = torch.split(model_return, 1, dim=1)
    return u, v, w, p


def train(pde_parameters, train_parameters, nn_parameters):
    # loading parameters
    [nf, opt, max_iter, min_loss, learning_rate, process] = train_parameters
    [lam, x_range] = pde_parameters
    # choose device to train model
    device = 'cpu'
    # x t input
    x_i = torch.FloatTensor(np.array(x_range[0]).reshape(-1, 1)).to(device)
    x_all = np.linspace(x_range[0], x_range[1], 10001).reshape(-1, 1)
    idx = np.random.choice(np.shape(x_all)[0], nf)
    x_star = x_all[idx, :]
    u_star = x_star + np.exp(x_star)
    v_star = x_star - np.exp(x_star)
    training_points = np.append(np.append(x_star, u_star, axis=1), v_star, axis=1)
    noise = 0.00
    normal_data1 = np.random.normal(0, 1, u_star.shape)
    normal_data2 = np.random.normal(0, 1, u_star.shape)
    u_star = u_star * (1 + noise * normal_data1)
    v_star = v_star * (1 + noise * normal_data2)
    x_f = torch.FloatTensor(x_star).to(device)
    u_f = torch.FloatTensor(u_star).to(device)
    v_f = torch.FloatTensor(v_star).to(device)
    # dynamic display during training
    x_test = np.linspace(x_range[0], x_range[1], 100)
    x_test_ts = torch.FloatTensor(x_test.flatten()[:, None]).to(device)
    fx_exact = x_test + np.exp(x_test)
    gx_exact = x_test - np.exp(x_test)
    # loading initial condition
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
        optimizer = torch.optim.Adam([{'params': net_model.parameters()}], lr=learning_rate)  # 设置模型的优化算法 及学习率
    elif opt == 2:
        optimizer = torch.optim.SGD([{'params': net_model.parameters()}], lr=learning_rate)  # 设置模型的优化算法 及学习率
    else:
        optimizer = torch.optim.LBFGS([{'params': net_model.parameters()}], lr=learning_rate)  # 设置模型的优化算法 及学习率
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
            f_i, g_i, v_i, w_i = pred(net_model, x_i, x_range)
            f_pred, g_pred, v_pred, w_pred = pred(net_model, x_f, x_range)

            f_pred_dx = torch.autograd.grad(f_pred.sum(), x_f, create_graph=True)[0]
            f_pred_dxx = torch.autograd.grad(f_pred_dx.sum(), x_f, create_graph=True)[0]
            g_pred_dx = torch.autograd.grad(g_pred.sum(), x_f, create_graph=True)[0]
            g_pred_dxx = torch.autograd.grad(g_pred_dx.sum(), x_f, create_graph=True)[0]
            v_pred_dx = torch.autograd.grad(v_pred.sum(), x_f, create_graph=True)[0]
            w_pred_dx = torch.autograd.grad(w_pred.sum(), x_f, create_graph=True)[0]

            f1 = 1.0 - 1 / 3 * x_f ** 3 - 1 / 2 * g_pred_dx ** 2 + lam_f[0] * v_pred - f_pred_dxx
            f2 = -1.0 + x_f ** 2 - x_f * f_pred + lam_f[1] * w_pred - g_pred_dxx
            e_outputs1 = v_pred_dx - f_pred ** 2 - g_pred ** 2
            e_outputs2 = w_pred_dx - f_pred ** 2 + g_pred ** 2
            e_f = f_pred - u_f
            e_g = g_pred - v_f
            loss_u_f = loss_func(e_f, torch.zeros_like(x_f)) + loss_func(e_g, torch.zeros_like(x_f))
            loss_i = loss_func(v_i, torch.zeros_like(x_i)) + loss_func(w_i, torch.zeros_like(x_i))
            loss_f = loss_func(f1, torch.zeros_like(x_f)) + loss_func(f2, torch.zeros_like(x_f))
            loss_o = loss_func(e_outputs1, torch.zeros_like(x_f)) + loss_func(e_outputs2, torch.zeros_like(x_f))
            loss_all = np.array((loss_i.data.numpy(), loss_f.data.numpy(), loss_o.data.numpy()))
            loss_min = np.min(loss_all)
            loss_all = loss_all / loss_min
            loss_all[loss_all > 10] = 10.0
            loss_weight = loss_all
            loss_total = 1.0 * loss_u_f + loss_weight[0] * loss_i + loss_weight[1] * loss_f + loss_weight[2] * loss_o
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
                fx_test, gx_test, v_test, w_test = pred(net_model, x_test_ts, x_range)
                fx_test = fx_test.cpu().data.numpy()
                gx_test = gx_test.cpu().data.numpy()
                plt.plot(x_test, fx_test, label='f Pred value')
                plt.plot(x_test, fx_exact, label='f Exact value')
                plt.plot(x_test, gx_test, label='g Pred value')
                plt.plot(x_test, gx_exact, label='g Exact value')
                plt.legend(loc='upper right')
                plt.pause(0.1)
        it = it + 1
    endtime_train = datetime.datetime.now()
    print('---------------End training:{}---------------'.format(endtime_train))
    torch.save(net_model, './model/A-PINN_D_VIDE-2D.pkl')
    np.savetxt('./data/A-PINN_D_VIDE-2D-loss.txt', loss_record)
    np.savetxt('./data/A-PINN_D_VIDE-2D-lam.txt', lam_record)
    np.savetxt('./data/training points(noise).txt', training_points)
    train_time = endtime_train - starttime_train
    print('---------------Training time:{}s---------------'.format(train_time.seconds + train_time.microseconds / 1e6))

    plt.ioff()
    plt.show()

    plt.figure()
    plt.plot(loss_record[:, 0], np.log(loss_record[:, 2]))
    plt.savefig('./png/A-PINN_D_VIDE-2D-loss.png')
    plt.show()


def test(pde_parameters, test_parameters):
    [_, x_range] = pde_parameters
    [dx] = test_parameters
    net_model = torch.load('./model/A-PINN_D_VIDE-2D.pkl')
    device = 'cpu'
    x_test = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) / dx)).reshape(-1, 1)
    x_test_ts = torch.FloatTensor(x_test.flatten()[:, None]).to(device)  # 网格拍平

    fx_test, gx_test, v_test, w_test = pred(net_model, x_test_ts, x_range)
    fx_test = fx_test.cpu().data.numpy()
    gx_test = gx_test.cpu().data.numpy()

    fx_exact = x_test + np.exp(x_test)
    gx_exact = x_test - np.exp(x_test)
    fgx_test = np.concatenate((fx_test, gx_test), axis=0)
    fgx_exact = np.concatenate((fx_exact, gx_exact), axis=0)
    l2error = np.linalg.norm(fgx_test - fgx_exact, 2) / np.linalg.norm(fgx_exact, 2)

    print('L2 error: ', l2error)
    plt.figure()
    plt.plot(x_test, fx_test, label='u(x) of A-PINN')
    plt.scatter(x_test, fx_exact, label='Exact u(x)', s=30, c='k', marker='x')
    plt.plot(x_test, gx_test, label='v(x) of A-PINN')
    plt.scatter(x_test, gx_exact, label='Exact v(x)', s=30, c='b', marker='x')
    plt.legend(loc='upper left')
    plt.xlim(x_range[0], x_range[1])
    plt.xlabel('x')
    plt.ylabel('u (x) and v (x)')
    plt.savefig('./png/A-PINN_D_VIDE-2D-u(x).png')
    plt.show()


if __name__ == '__main__':
    main()
    pass
