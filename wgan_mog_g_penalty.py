from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.tensor as Variable
import torch.autograd as ag

class mog_gen:
    def __init__(self, dim):
        self.theta = 1.2
        self.mu_list = [0, 4.5, 8.2]
        self.sigma_list = [0.05, 0.5, 0.25]
        self.dim   = dim
    def get_random_sample(self, batch_size):
        sample_list = []
        for samp in range(batch_size):
            mode_sel = np.random.rand(1)
            x_g = torch.randn(1).cuda().float()
            if(mode_sel < 0.2):
                x = x_g*self.sigma_list[0] + self.mu_list[0]
            elif(mode_sel < 0.7):
                x = x_g*self.sigma_list[1] + self.mu_list[1]
            else:
                x = x_g*self.sigma_list[2] + self.mu_list[2]
            sample = np.array([self.theta, x])
            sample_list.append(sample)
        samples = np.array(sample_list)
        return samples

class g_func(nn.Module):
    def __init__(self, e_dim, in_dim, out_dim):
        super(g_func, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = e_dim
        self.out_dim = out_dim
        self.fc_1    = nn.Linear(self.in_dim, self.hidden_dim)
        self.relu_1  = nn.LeakyReLU()
        self.fc_2    = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_2  = nn.LeakyReLU()
        self.fc_3    = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_3  = nn.LeakyReLU()
        self.fc_4    = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_4  = nn.LeakyReLU()
        self.fc_5    = nn.Linear(self.hidden_dim, self.out_dim)
    def forward(self, z):
        x = self.fc_1(z)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        x = self.relu_3(x)
        x = self.fc_4(x)
        x = self.relu_4(x)
        x = self.fc_5(x)
        return x

class f_func(nn.Module):
    def __init__(self, e_dim, in_dim, m):
        super(f_func, self).__init__()
        self.in_dim     = in_dim*m
        self.hidden_dim = e_dim
        self.out_dim    = 1
        self.fc_1       = nn.Linear(self.in_dim, self.hidden_dim)
        self.relu_1     = nn.LeakyReLU()
        self.fc_2       = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_2     = nn.LeakyReLU()
        self.fc_3       = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_3     = nn.LeakyReLU()
        self.fc_4       = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu_4     = nn.LeakyReLU()
        self.fc_5       = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        dim        = x.size()[1]
        in_vec     = torch.zeros(int(batch_size*0.5), int(dim*2)).cuda().float()
        in_vec[:,:dim] = x[:int(batch_size*0.5),:]
        in_vec[:, dim:] = x[int(batch_size*0.5):,:]
        z = in_vec
        x = self.fc_1(z)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        x = self.relu_3(x)
        x = self.fc_4(x)
        x = self.relu_4(x)
        x = self.fc_5(x)
        return x
    def clip_weights(self, c):
        self.fc_1.weight.data = torch.clamp(self.fc_1.weight.data, -c, c)
        self.fc_2.weight.data = torch.clamp(self.fc_2.weight.data, -c, c)
        self.fc_3.weight.data = torch.clamp(self.fc_3.weight.data, -c, c)


m           = 2    # Degree of packing.
d           = 2    # Dimension of the real data distribution.
e_dim       = 1024 # Dimension of the hidden layers.
f           = f_func(e_dim, d, m).cuda()
g           = g_func(e_dim, d, d).cuda()
# Load the previous check-point.
f.load_state_dict(torch.load('wgan_critic.ckpt'))
g.load_state_dict(torch.load('wgan_gen.ckpt'))

f_opt       = optim.Adam(f.parameters(), lr=1e-3, weight_decay=1e-8)
g_opt       = optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-8)
f_scheduler = optim.lr_scheduler.StepLR(f_opt, step_size=5000, gamma=0.1)
g_scheduler = optim.lr_scheduler.StepLR(g_opt, step_size=1000, gamma=0.1)

# Gradient Penalty Hyper-parameters.
c = 1e-2
batch_size = 256
lmbda = 10

max_iters = 1000
sample_gen = mog_gen(d)

# This loop implements the gradient penalty and Pac-GAN learning algorithm.
for it in range(max_iters):
    for t_critic in range(5):
        data  = sample_gen.get_random_sample(batch_size)
        x     = Variable(torch.from_numpy(data).cuda().float(), requires_grad = True)
        z     = Variable(torch.randn(batch_size,d), requires_grad=True).cuda().float().mul(1)
        f_x   = f(x)
        g_z   = g(z)
        fg_z = f(g_z)
        eps   = Variable(torch.rand(batch_size), requires_grad=True).cuda().float()
        x1 = Variable(torch.matmul(torch.diag(eps), x), requires_grad=True)
        x2 = Variable(torch.matmul(torch.diag(1-eps), g_z), requires_grad=True)
        x_hat = Variable(x1 + x2, requires_grad= True)
        f_xh  = f(x_hat)
        grad_xh_norm = torch.zeros(batch_size).cuda().float()
        for b in range(f_xh.size()[0]):
            g_x_hat = ag.grad(f_xh[b][0], x_hat, retain_graph=True)[0]
            grad_xh_norm[b] = torch.norm(g_x_hat)
        #grad_xh_norm = torch.zeros(batch_size).cuda().float()
        #for b in range(batch_size):
        #    x_hat = Variable(x[b,:].mul(eps[b]) + g_z[b,:].mul(1-eps[b]), requires_grad= True)
        #    f_xh  = f(x_hat)
        #    g_x_hat = ag.grad(f_xh, x_hat, retain_graph=True)[0]
        #    grad_xh_norm[b] = torch.norm(g_x_hat)
        f_loss = torch.sum(fg_z - f_x + (grad_xh_norm-1)*(grad_xh_norm-1)*lmbda).div(batch_size)
        f_opt.zero_grad()
        f_loss.backward()
        f_scheduler.step()
        f_opt.step()
    z = torch.randn(batch_size,2).cuda().float().mul(1)
    fg_z = f(g(z)).mul(-1)
    g_loss = torch.sum(fg_z).div(batch_size)
    g_opt.zero_grad()
    g_loss.backward()
    g_scheduler.step()
    g_opt.step()
    if((it+1)%20 == 0):
        print(str(it)+':   F loss: ' + str(f_loss) + ',   G loss: ' + str(g_loss))
    if((it+1)%(max_iters/20) == 0):
        data = sample_gen.get_random_sample(batch_size)
        z = torch.randn(batch_size,2).cuda().float().mul(1)
        gen_data = g(z).detach().cpu().numpy()
        z_data = z.detach().cpu().numpy()
        plt.figure()
        plt.scatter(z_data[:,0],   z_data[:,1],   color='green')
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(gen_data[:,0], gen_data[:,1], color='red')
        plot_name = 'plt_'+str(it)+'.png'
        plt.savefig(plot_name)
        torch.save(f.state_dict(), 'wgan_critic.ckpt')
        torch.save(g.state_dict(), 'wgan_gen.ckpt')
