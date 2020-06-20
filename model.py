import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Model(nn.Module):
    def __init__(self, stats):
        super().__init__()
        input_dim = 40
        inter_dim = 128
        self.num_layers = 2
        self.l1 = nn.Linear(input_dim, inter_dim)
        self.inter = nn.ModuleList(
                [nn.Linear(inter_dim,inter_dim) for _ in range(self.num_layers)]
        )
        self.l2 = nn.Linear(inter_dim, 4)

        self.stat_mean = nn.Parameter(torch.Tensor(stats[0]), requires_grad=False)
        self.stat_std = nn.Parameter(torch.Tensor(stats[1]), requires_grad=False)

        self.dynm_mean = nn.Parameter(torch.Tensor(stats[2]), requires_grad=False)
        self.dynm_std = nn.Parameter(torch.Tensor(stats[3]), requires_grad=False)


    def forward(self, data, dynamic):
        data = (data - self.stat_mean) / self.stat_std
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

        # ======= linear ======
        input_dim = 40
        inter_dim = 128
        self.num_layers = 2
        self.l1 = nn.Linear(input_dim, inter_dim)
        self.inter = nn.ModuleList(
                [nn.Linear(inter_dim,inter_dim) for _ in range(self.num_layers)]
        )
        self.l2 = nn.Linear(inter_dim, 4)
        
        self.stat_mean = nn.Parameter(torch.Tensor(stats[0]), requires_grad=False)
        self.stat_std = nn.Parameter(torch.Tensor(stats[1]), requires_grad=False)

        # ======= rnn ======
        rnn_input_dim = 8
        rnn_inter_dim = 64
        self.pre_rnn = nn.Linear(rnn_input_dim, rnn_inter_dim)
        self.rnn = nn.LSTM(rnn_inter_dim, rnn_inter_dim, batch_first=True, num_layers=2)
        self.post_rnn = nn.Linear(rnn_inter_dim, 4)

        self.dynm_mean = nn.Parameter(torch.Tensor(stats[2]), requires_grad=False)
        self.dynm_std = nn.Parameter(torch.Tensor(stats[3]), requires_grad=False)


    def forward(self, data, dynamic):
        # static
        data = (data - self.stat_mean) / self.stat_std
        out = F.relu(self.l1(data))# + torch.tanh(self.l1_2(data**2))

        for _l in range(self.num_layers):
            prev = out
            out = self.inter[_l](out)
            out = F.relu(out) + prev
        
        linear_out = self.l2(out)
        
        # dynamic
        input_length = ((dynamic==0).sum(-1) == 0).sum(1)
        data = (dynamic - self.dynm_mean) / self.dynm_std
        data = self.pre_rnn(data)
        packed = nn.utils.rnn.pack_padded_sequence(data, input_length,
                enforce_sorted=False, batch_first=True)
        _, out = self.rnn(packed)
        h, c = out

        dynamic_out = self.post_rnn(h[-1])

        out = (linear_out + dynamic_out) / 2.
        return out

    def get_loss(self, pred, target):

        e1= torch.mean(
            torch.sum((pred[:,:2] - target[:,:2])**2, 1) / 2e+4
        )
        e2 = torch.mean(
            torch.sum(((pred[:,2:] - target[:,2:]) / (target[:,2:]+1e-6))**2, 1)
        )
        return (0.5*e1 + 0.5*e2).mean()
