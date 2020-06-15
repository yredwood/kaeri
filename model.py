import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class Model(nn.Module):
    def __init__(self, stats):
        super().__init__()
        input_dim = 28 + 4 * 8
        self.num_layers = 2
        self.l1 = nn.Linear(input_dim, 128)
        self.inter = nn.ModuleList(
                [nn.Linear(128,128) for _ in range(self.num_layers)]
        )
        self.l2 = nn.Linear(128, 4)
        #self.bn_list = nn.ModuleList(
                #[nn.BatchNorm1d(32) for _ in range(self.num_layers)]
        #)
        #self.prebn = nn.BatchNorm1d(input_dim, affine=False)
#        self.feat_mean = nn.Parameter(\
#                torch.Tensor([106, 106, 105, 82, 8e+5, 8.4e+5, 8.4e+5, 1e+6,
#            10, 10, 10, 10]))
#
#        self.feat_std = nn.Parameter(\
#                torch.Tensor([40, 40, 34, 33,
#                    6.6e+5, 6.5e+5, 5.8e+5, 8e+5,
#                    11, 11, 2.9, 3.0]))

        self.stat_mean = nn.Parameter(torch.Tensor(stats[0]), requires_grad=False)
        self.stat_std = nn.Parameter(torch.Tensor(stats[1]), requires_grad=False)

        #self.dyn_mean = nn.Parameter(torch.Tensor(stats[2]))


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
