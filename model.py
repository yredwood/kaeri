import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class Model(nn.Module):
    def __init__(self, stats):
        super().__init__()
        input_dim = 40
        self.num_layers = 2
        self.l1 = nn.Linear(input_dim, 128)
        self.inter = nn.ModuleList(
                [nn.Linear(128,128) for _ in range(self.num_layers)]
        )
        self.l2 = nn.Linear(128, 4)

        self.stat_mean = nn.Parameter(torch.Tensor(stats[0]), requires_grad=False)
        self.stat_std = nn.Parameter(torch.Tensor(stats[1]), requires_grad=False)

        #self.dyn_mean = nn.Parameter(torch.Tensor(stats[2]))

        # inverse params
#        self.inv_l1 = nn.Linear(2, 128)
#        self.inv_l2 = nn.Linear(128, 1)
#        self.sc = nn.Parameter(torch.ones(1))
##        self.xs = nn.Parameter(torch.zeros(4))
##        self.ys = nn.Parameter(torch.zeros(4))
#
#        self.xs = nn.Parameter(torch.Tensor([-5.2691, 5.2691, -0.0119, 0.0061]))
#        self.ys = nn.Parameter(torch.Tensor([-3.04, -3.04, 6.06, 4.05]))

    def forward(self, data, dynamic):
        data = (data - self.stat_mean) / self.stat_std
        #out = F.relu(self.l1(data))
        out = F.relu(self.l1(data))# + torch.tanh(self.l1_2(data**2))

        for _l in range(self.num_layers):
            prev = out
            out = self.inter[_l](out)
            out = F.relu(out) + prev
        
        out = self.l2(out)
        return out

    def get_loss(self, pred, target):

        e1= torch.mean(
            torch.sum((pred[:,:2] - target[:,:2])**2, 1) / 2e+4
        )
        e2 = torch.mean(
            torch.sum(((pred[:,2:] - target[:,2:]) / (target[:,2:]+1e-6))**2, 1)
        )
        return (0.5*e1 + 0.5*e2).mean()

    def invert_forward(self, data, dynamic):
        # data: (x,y,m,v)
        l = (data[:,0:1] / 100. - self.xs.unsqueeze(0))**2 + (data[:,1:2] / 100. - self.ys.unsqueeze(0))**2
        s = torch.sqrt(l) * self.sc # (b,4)
        
        input = torch.cat([s.unsqueeze(2), data[:,3:4].unsqueeze(1).repeat(1,4,1)], -1)

        out = F.relu(self.inv_l1(input))

#        for _l in range(self.num_layers):
#            prev = out
#            out = self.inter[_l](out)
#            out = F.relu(out) + prev

        out = self.inv_l2(out)
        t = out.squeeze(-1)

#        v = data[:,3:4] * self.vc
#        a = data[:,2:3] * self.a + 1e-5
#        t = torch.sqrt( 2 * s / a + (v / a)**2 ) - v / a

        #t = data[:,3:4] / 2 / self.a + torch.sqrt(s / self.a - (data[:,3:4] / self.a / 2)**2)
        #t = torch.sqrt( 2*s / self.a + data[:,3:4]**2 / self.a / 2 ) - data[:,3:4] / self.a + self.b
        
        return t

    def invert_get_loss(self, pred, target):
        # only compares distance
        # get first time feature
        time = target[:,:20].reshape(-1,5,4)[:,0]
        return torch.abs(pred - time).mean()



class RNN(nn.Module):
    def __init__(self, stats):
        super().__init__()
        static_dim = 12
        self.num_layers = 2
        self.rnn_hdim = 32

        self.l1 = nn.Linear(static_dim,32)
        self.inter = nn.ModuleList(
                [nn.Linear(32,32) for _ in range(self.num_layers)]
        )
        self.stat_mean = nn.Parameter(torch.Tensor(stats[0]), requires_grad=True)
        self.stat_std = nn.Parameter(torch.Tensor(stats[1]), requires_grad=True)

        self.dynm_mean = nn.Parameter(torch.Tensor(stats[2]), requires_grad=True)
        self.dynm_std = nn.Parameter(torch.Tensor(stats[3]), requires_grad=True)
        
        self.pre_rnn = nn.Linear(4, self.rnn_hdim)
        self.rnn = nn.LSTM(self.rnn_hdim, self.rnn_hdim, batch_first=True, num_layers=2)
        self.dense_last = nn.Linear(32 + self.rnn_hdim, 4)


    def forward(self, static, dynamic):

        static = (static - self.stat_mean) / self.stat_std
        out = self.l1(static)

        for _l in range(self.num_layers):
            prev = out
            out = self.inter[_l](out)
            out = F.relu(out) + prev

        static_feat = out
        
        dynamic = (dynamic - self.dynm_mean) / self.dynm_std
        out = self.pre_rnn(dynamic)
        _, dynamic_feat = self.rnn(out)
        dynamic_feat = dynamic_feat[0][-1]

        feat = torch.cat([static_feat, dynamic_feat], -1)
        out = self.dense_last(feat)
        
        return out

    def get_loss(self, pred, target):

        e1= torch.mean(
            torch.sum((pred[:,:2] - target[:,:2])**2, 1) / 2e+4
        )
        e2 = torch.mean(
            torch.sum(((pred[:,2:] - target[:,2:]) / (target[:,2:]+1e-6))**2, 1)
        )
        return (0.5*e1 + 0.5*e2).mean()
