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
    t_range = [0.0, 1.0]
    pde_parameters = [lam, x_range, t_range]

    # train parameters
    ni = 100
    nb = 100
    nf = 5000
    optimizer = 0  # 0: L-BFGS 1: Adam 2: SGD
    max_iter = 500
    min_loss = 1e-8
    learning_rate = 0.01
    process = True
    train_parameters = [ni, nb, nf, optimizer, max_iter, min_loss, learning_rate, process]

    # test parameters
    dx = 0.01
    dt = 0.01
    test_parameters = [dx, dt]

    # Neural networks parameters
    nn_layers = [2, 40, 40, 3]  # neural networks layers
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
        for i in range(len(nn_layers)-2):
            self.layers.append(torch.nn.Linear(nn_layers[i], nn_layers[i + 1]),)
            self.layers.append(activation_function)
        self.layers.append(torch.nn.Linear(nn_layers[-2], nn_layers[-1]),)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class Sin(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.sin(x)


def pred(net_model, x, t, x_range, t_range):
    x_lb = x_range[0]
    x_ub = x_range[1]
    x_ = 2.0 * (x - x_lb) / (x_ub - x_lb) - 1.0
    t_lb = t_range[0]
    t_ub = t_range[1]
    t_ = 2.0 * (t - t_lb) / (t_ub - t_lb) - 1.0
    model_return = net_model(torch.cat((x_, t_), 1))
    u, v, w = torch.split(model_return, 1, dim=1)
    return u, v, w


def train(pde_parameters, train_parameters, nn_parameters):
    # loading parameters
    [ni, nb, nf, opt, max_iter, min_loss, learning_rate, process] = train_parameters
    [_, x_range, t_range] = pde_parameters
    # choose device to train model
    device = 'cpu'
    # x t input
    t_i = torch.FloatTensor(np.ones([ni, 1]) * t_range[0]).to(device)  # 初始t值
    x_i = torch.FloatTensor(np.linspace(x_range[0], x_range[1], ni, endpoint=True).reshape(-1, 1)).to(device)
    t_lb = torch.FloatTensor(np.linspace(t_range[0], t_range[1], int(nb / 2), endpoint=True).reshape(-1, 1)).to(device)
    x_lb = torch.FloatTensor(np.ones([int(nb / 2), 1]).reshape(-1, 1) * x_range[0]).to(device)  # 下边界

    x_exact = np.arange(x_range[0], x_range[1] + 1e-3, 1e-3)
    t_exact = np.arange(t_range[0], t_range[1] + 1e-3, 1e-3)
    x_e_mesh, t_e_mesh = np.meshgrid(x_exact, t_exact)
    t_e = t_e_mesh.flatten()
    x_e = x_e_mesh.flatten()
    idx = np.random.choice(x_e.shape[0], nf, replace=False)
    t_star = t_e[idx].reshape(-1, 1)
    t_f = torch.FloatTensor(t_star).to(device)
    x_star = x_e[idx].reshape(-1, 1)
    x_f = torch.FloatTensor(x_star).to(device)

    # dynamic display during training
    x_test = np.linspace(x_range[0], x_range[1], 100)
    t_test = np.linspace(t_range[0], t_range[1], 100)
    t_mesh, x_mesh = np.meshgrid(t_test, x_test)
    tx_shape = np.shape(x_mesh)
    x_test_ts = torch.FloatTensor(x_mesh.flatten()[:, None]).to(device)
    t_test_ts = torch.FloatTensor(t_mesh.flatten()[:, None]).to(device)
    u_test = x_test_ts + t_test_ts * torch.sin(t_test_ts + x_test_ts)
    u_test = u_test.data.numpy().reshape(tx_shape)
    # loading initial condition
    u_i = x_i
    u_i_dt = torch.sin(x_i)
    # set x t require grad
    t_i.requires_grad = True
    t_lb.requires_grad = True
    x_lb.requires_grad = True
    t_f.requires_grad = True
    x_f.requires_grad = True
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
    plt.ion()
    print('------------------------Neural networks------------------------------------')
    print(net_model)
    print('----------------------------Optimizer--------------------------------------')
    print(optimizer)
    print('------------------------Training device: ', device, '-----------------------------')
    #  -----------   start training  ------------
    starttime_train = datetime.datetime.now()
    print('------------------------Start training:{}---------------------'.format(starttime_train))

    while it < max_iter and loss > min_loss:
        def closure():
            u_i_pred, v_i_pred, w_i_pred = pred(net_model, x_i, t_i, x_range, t_range)
            u_lb_pred, v_lb_pred, w_lb_pred = pred(net_model, x_lb, t_lb, x_range, t_range)
            u_f_pred, v_f_pred, w_f_pred = pred(net_model, x_f, t_f, x_range, t_range)

            u_i_pred_dt = torch.autograd.grad(u_i_pred.sum(), t_i, create_graph=True)[0]
            u_f_pred_dt = torch.autograd.grad(u_f_pred.sum(), t_f, create_graph=True)[0]
            u_f_pred_dtt = torch.autograd.grad(u_f_pred_dt.sum(), t_f, create_graph=True)[0]
            u_f_pred_dx = torch.autograd.grad(u_f_pred.sum(), x_f, create_graph=True)[0]

            v_f_pred_dx = torch.autograd.grad(v_f_pred.sum(), x_f, create_graph=True)[0]
            w_f_pred_dt = torch.autograd.grad(w_f_pred.sum(), t_f, create_graph=True)[0]

            e_boundary1 = u_lb_pred - t_lb * torch.sin(t_lb)

            g_f = t_f * (-t_f ** 2 * torch.sin(t_f) ** 2 / 4 + torch.sin(t_f) + torch.sin(t_f) ** 2 / 8) + t_f *\
                  (x_f * torch.cos(x_f)- torch.sin(x_f) ** 2 / 8 - torch.sin(x_f)) - t_f * (-t_f ** 2 *
                  torch.sin(t_f - x_f) * torch.sin(t_f + x_f) / 4 + t_f * x_f * torch.sin(t_f - x_f) *
                   torch.sin(t_f + x_f) / 4 - t_f * x_f * torch.cos(t_f - x_f) * torch.cos(t_f + x_f) / 4 + x_f
                   * torch.sin(t_f - x_f) * torch.cos(t_f + x_f) / 8 + x_f * torch.sin(t_f + x_f) *
                   torch.cos(t_f - x_f) / 8 + x_f * torch.cos(t_f - x_f) + torch.sin(t_f - x_f) +
                   torch.sin(t_f - x_f) * torch.sin(t_f + x_f) / 8) + x_f + torch.sin(t_f + x_f) + 2 * torch.cos(t_f + x_f) - 1
            f = u_f_pred_dx - u_f_pred_dt - u_f_pred + g_f + v_f_pred - u_f_pred_dtt
            e_outputs1 = v_f_pred_dx - t_f * w_f_pred
            e_outputs2 = w_f_pred_dt - torch.cos(t_f-x_f) * u_f_pred

            loss_0 = loss_func(u_i_pred, u_i) + loss_func(u_i_pred_dt, u_i_dt) + loss_func(w_i_pred, torch.zeros_like(t_i))
            loss_b = loss_func(e_boundary1, torch.zeros_like(t_lb)) + loss_func(v_lb_pred, torch.zeros_like(t_lb))
            loss_f = loss_func(f, torch.zeros_like(t_f))
            loss_o = loss_func(e_outputs1, torch.zeros_like(t_f)) + loss_func(e_outputs2, torch.zeros_like(t_f))

            loss_all = np.array((loss_0.data.numpy(), loss_b.data.numpy(), loss_f.data.numpy(), loss_o.data.numpy()))
            loss_min = np.min(loss_all)
            loss_all = loss_all / loss_min
            loss_all[loss_all > 100] = 100.0
            loss_weight = loss_all

            loss_total = loss_weight[0] * loss_0 + loss_weight[1] * loss_b + loss_weight[2] * loss_f + loss_weight[3] * loss_o
            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            return loss_total

        optimizer.step(closure)
        loss_value = closure().cpu().data.numpy()
        step_time = datetime.datetime.now() - starttime_train
        loss_record = np.append(loss_record, [[it, step_time.seconds + step_time.microseconds / 1000000, loss_value]],
                                axis=0)

        if it % 10 == 0:
            print('Running: ', it, ' / ', max_iter)
            if process:
                plt.clf()
                u_test_pred, _, _ = pred(net_model, x_test_ts, t_test_ts, x_range, t_range)
                u_test_pred = u_test_pred.data.numpy().reshape(tx_shape)
                pic = plt.imshow(u_test_pred, cmap='jet', vmin=-1.0, vmax=1.0,
                                 extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                                 interpolation='nearest', origin='lower', aspect='auto')
                plt.colorbar(pic)
                plt.scatter(t_star, x_star, label='Collocation points', s=1.0, c='k', marker='.')
                plt.xlim(t_range[0], t_range[1])
                plt.ylim(x_range[0], x_range[1])
                plt.legend(loc='upper right')
                plt.pause(0.1)
        it = it + 1
    endtime_train = datetime.datetime.now()
    print('---------------End training:{}---------------'.format(endtime_train))
    torch.save(net_model, './model/A-PINN_S_VIDE.pkl')
    np.savetxt('./data/A-PINN_S_VIDE-loss.txt', loss_record)

    train_time = endtime_train - starttime_train  
    print('---------------Training time:{}s---------------'.format(train_time.seconds + train_time.microseconds / 1e6))

    plt.ioff()
    plt.show()

    plt.figure()
    plt.plot(loss_record[:, 0], np.log(loss_record[:, 2]))
    plt.savefig('./png/A-PINN_S_VIDE-loss.png')
    plt.show()


def test(pde_parameters, _):

    x_exact = np.arange(0, 1 + 0.01, 0.01)
    t_exact = np.arange(0, 1 + 1e-4, 1e-4)
    t_test = t_exact.flatten()
    x_test = x_exact.flatten()

    [_, x_range, t_range] = pde_parameters
    net_model = torch.load('./model/A-PINN_S_VIDE.pkl')
    device = 'cpu'

    x_mesh, t_mesh = np.meshgrid(x_test, t_test)
    xt_shape = np.shape(x_mesh)
    t_test_ts = torch.FloatTensor(t_mesh.flatten()[:, None]).to(device)
    x_test_ts = torch.FloatTensor(x_mesh.flatten()[:, None]).to(device)
    u_test = x_test_ts + t_test_ts * torch.sin(t_test_ts + x_test_ts)
    u_exact = u_test.data.numpy().reshape(xt_shape).T

    u_test_pred, _, _ = pred(net_model, x_test_ts, t_test_ts, x_range, t_range)
    u_test_pred = u_test_pred.data.numpy().reshape(xt_shape).T

    l2error = np.linalg.norm(u_test_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
    abs_e = u_test_pred - u_exact
    print('L2 error: ', l2error)

    plt.figure(figsize=(8, 3))
    pic = plt.imshow(u_test_pred, cmap='jet', vmin=0.0, vmax=2.0,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('u(t,x)')
    plt.tight_layout()
    plt.savefig('./png/A-PINN_S_VIDE-u(x,t).png')
    plt.show()
    plt.figure(figsize=(8, 3))
    pic = plt.imshow(abs_e, cmap='jet', vmin=-0.01, vmax=0.01,
                     extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                     interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(pic)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Error in u(t,x)')
    plt.tight_layout()
    plt.savefig('./png/A-PINN_S_VIDE-abs(error).png')
    plt.show()


if __name__ == '__main__':
    main()
    pass
