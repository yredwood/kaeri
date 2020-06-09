import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.l1 = nn.Linear(12,32)
        self.inter = nn.ModuleList(
                [nn.Linear(32,32) for _ in range(self.num_layers)]
        )
        #self.bn_list = nn.ModuleList(
                #[nn.BatchNorm1d(32) for _ in range(self.num_layers)]
        #)
        self.l2 = nn.Linear(32, 4)
        self.feat_mean = nn.Parameter(\
                torch.Tensor([106, 106, 105, 82, 8e+5, 8.4e+5, 8.4e+5, 1e+6,
            10, 10, 10, 10]))

        self.feat_std = nn.Parameter(\
                torch.Tensor([40, 40, 34, 33,
                    6.6e+5, 6.5e+5, 5.8e+5, 8e+5,
                    11, 11, 2.9, 3.0]))


    def forward(self, data):
        data = (data - self.feat_mean) / self.feat_std
        out = self.l1(data)

        for _l in range(self.num_layers):
            out = self.inter[_l](out)
            #out = self.bn_list[_l](out)
            out = F.relu(out) + out
        
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
