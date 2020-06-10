
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import PackedSequence

import pdb

class Block(nn.Module):
    def __init__(self, n_in, n_out, shortcut=False):
        super().__init__()

        filter_size = 3
        padding = filter_size // 2
        self.conv1 = nn.Conv2d(n_in, n_out, filter_size, stride=2, padding=padding, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class ConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 32
        self.input_dim = 4
        
        layers = [Block(1, 32), Block(32, 32)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1, self.input_dim)
        x = self.backbone(x) # b, 32, T//4, 80//4
        x = x.permute(0,2,1,3)
        x = x.reshape(x.size(0), x.size(1), -1) # b,T//4, 32*20
        x = x.transpose(0,1)
        return x


class LSTM_BN(nn.Module):
    def __init__(self, input_size, hidden_size, shortcut, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        self.shortcut = shortcut

        self.rnn_list = nn.ModuleList([
            nn.LSTM(input_size if layers==0 else hidden_size*2,
                hidden_size, num_layers=1, bidirectional=True)
            for layers in range(self.num_layers)])

        self.dense_list = nn.ModuleList([
            nn.Linear(self.hidden_size*2, self.hidden_size*2)
            for layer in range(self.num_layers-1)])

        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size*2) 
            for layer in range(self.num_layers-1)])

    def forward(self, input, hidden):
        h, c = hidden
        def reshape_hidden(x):
            nl2, bsz, hdim = x.size()
            x = x.reshape(self.num_layers,2,bsz,hdim)
            return x
        h = reshape_hidden(h)
        c = reshape_hidden(c)

        hlist, clist = [], []
        for i, rnn in enumerate(self.rnn_list):
            residual = input.data
            output, (ho, co) = rnn(input, (h[i], c[i]))
            hlist.append(ho)
            clist.append(co)
            
            input = output.data
            if i!=self.num_layers-1:
                input = self.dense_list[i](input)
                input = self.bn_list[i](input)
                input = F.relu(input)

            if i > 0 and self.shortcut:
                input = input + residual

            input = get_packed_sequence(
                    data=input, batch_sizes=output.batch_sizes,
                    sorted_indices=output.sorted_indices,
                    unsorted_indices=output.unsorted_indices)

        hlist = torch.cat(hlist, 0)
        clist = torch.cat(clist, 0)
        return input, (hlist, clist)


class GST(nn.Module):
    def __init__(self, output_dim, num_layers=4):
        '''
        las-style encoder
        '''
        super().__init__()
        
        input_dim = 4
        
        self.pre_conv = ConvEmbedding()
        self.preconv_dim = 32 * self.pooling(input_dim)
        
        self.lstm_hidden_dim = output_dim // 2 
        self.lstm_num_layers = num_layers
        self.lstm = LSTM_BN(
                input_size=self.preconv_dim,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                shortcut=True,
        )
        self.output_dim = self.lstm_hidden_dim * 2


    def forward(self, x):
        
        x_len = self.calc_length(x)
        x_emb = self.pre_conv(x) # (T,bsz,640)
        #x_emb = F.dropout(x_emb, p=0.5, training=self.training)

        pooled_length = [self.pooling(_l) for _l in x_len]
        pooled_length = x_emb.new_tensor(pooled_length).long()
        #assert pooled_length[0] == x_emb.size(0)

        state_size = self.lstm_num_layers*2, x_emb.size(1), self.lstm_hidden_dim
        fw_x = nn.utils.rnn.pack_padded_sequence(x_emb, pooled_length, enforce_sorted=False)
        fw_h = x_emb.new_zeros(*state_size)
        fw_c = x_emb.new_zeros(*state_size)
        packed_outputs, (final_hiddens, final_cells) = self.lstm(fw_x, (fw_h, fw_c))

        # not using final_h, final_c
#        final_outs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=.0)
#        #final_outs = F.dropout(final_outs, p=0.5, training=self.training)
#        return final_outs.transpose(0,1), pooled_length

        # using final_h
        final_outs = final_hiddens.view(-1,2,final_hiddens.size(-2),final_hiddens.size(-1))
        final_outs = torch.cat((final_outs[-1,0], final_outs[-1,1]), dim=-1)

        return final_outs.unsqueeze(1)


    def pooling(self, x):
        for _ in range(len(self.pre_conv.backbone)):
            #x = (x - 3 + 2 * 3//2) // 2 + 1
            x = x // 2 
        return x

    def calc_length(self, x):
        x_len = [x.size(-1) for _ in range(x.size(0))]
        for t in reversed(range(x.size(-1))):
            pads = (x[:,:,t].sum(1) == 0).int().tolist()
            x_len = [x_len[i] - pads[i] for i in range(len(x_len))]

            if sum(pads) == 0:
                break
        return x_len

def get_packed_sequence(data, batch_sizes, sorted_indices, unsorted_indices):
        return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


class RNN_1(nn.Module):

    def __init__(self, label_stats):
        super().__init__()

        self.hidden_size = 512

        self.dense1 = nn.Linear(4, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.dense3 = nn.Linear(self.hidden_size + 4*4, 4)
        
        with open(label_stats, 'r') as f: 
            lines = f.readlines()
        self.M = torch.FloatTensor([float(_b) for _b in lines[1].split(',')]).cuda()
        self.S = torch.FloatTensor([float(_b) for _b in lines[2].split(',')]).cuda()


    def forward(self, x):
        x = x[:,:,1:] # not use exact time feature

        # dynamic features
        out = self.dense1( x[:,:30] )         
        out = F.relu(out)
        out = self.dense2( out ) # not use exact time feature
        out = F.relu(out)
        _, out = self.rnn(out)
        out = out[0].squeeze(0)


        # static features
        _min, _ = torch.min(x, 1) # bsz,4
        _max, _ = torch.max(x, 1)
        _mean = torch.mean(x, 1)
        _std = torch.std(x, 1)
        
        static_feat = torch.cat(
            [out, _min, _max, _mean, _std], -1
        )

        out = self.dense3(static_feat)
        out = out * self.S.unsqueeze(0) + self.M.unsqueeze(0)
        return out

    
class Lin_1(nn.Module):

    def __init__(self, label_stats):
        super().__init__()

        self.hidden_size = 512

        self.dense_1 = nn.Linear(4*4, self.hidden_size)
        self.dense_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense_out = nn.Linear(self.hidden_size, 4)
        
        with open(label_stats, 'r') as f: 
            lines = f.readlines()
        self.M = torch.FloatTensor([float(_b) for _b in lines[1].split(',')]).cuda()
        self.S = torch.FloatTensor([float(_b) for _b in lines[2].split(',')]).cuda()


    def forward(self, x):
        x = x[:,:,1:] # not use exact time feature

        # static features
        _min, _ = torch.min(x, 1) # bsz,4
        _max, _ = torch.max(x, 1)
        _mean = torch.mean(x, 1)
        _std = torch.std(x, 1)
        
        static_feat = torch.cat(
            [ _min, _max, _mean, _std], -1
        )

        out = F.relu(self.dense_1(static_feat))
        out = F.relu(self.dense_2(out))
        out = self.dense_out(out)

        out = out * self.S.unsqueeze(0) + self.M.unsqueeze(0)
        return out

class RNN_3(nn.Module):

    def __init__(self, label_stats):
        super().__init__()

        self.hidden_size = 512
        
        # dynamic path
        self.dense_1 = nn.Linear(4, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=2)

        # static path
        self.dense_2 = nn.Linear(4*4, self.hidden_size)
        self.dense_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense_out = nn.Linear(self.hidden_size*2, 4)

        with open(label_stats, 'r') as f: 
            lines = f.readlines()
        self.M = torch.FloatTensor([float(_b) for _b in lines[1].split(',')]).cuda()
        self.S = torch.FloatTensor([float(_b) for _b in lines[2].split(',')]).cuda()


    def forward(self, x):
        x = x[:,:,1:] # not use exact time feature

        # dynamic features
        out = self.dense_1( x )
        _, out = self.rnn(out) # tuple (hidden, cell)
        dynamic_out = out[0][-1] # bsz, dim

        # static features
        _min, _ = torch.min(x, 1) # bsz,4
        _max, _ = torch.max(x, 1)
        _mean = torch.mean(x, 1)
        _std = torch.std(x, 1)
        
        static_feat = torch.cat(
            [_min, _max, _mean, _std], -1
        )

        out = self.dense_2(static_feat)
        out = F.relu(out)
        static_out = F.relu(self.dense_3(out))

        feat = torch.cat([dynamic_out, static_out], dim=-1) 
        out = self.dense_out(feat)
        out = out * self.S.unsqueeze(0) + self.M.unsqueeze(0)
        return out

class CONV_1(nn.Module):

    def __init__(self, label_stats):
        super().__init__()

        self.hidden_size = 512
        
        # dynamic path
        self.dense_1 = nn.Linear(4, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=2)

        # static path
        self.dense_2 = nn.Linear(4*4, self.hidden_size)
        self.dense_3 = nn.Linear(self.hidden_size, self.hidden_size)

        # conv path
        self.conv_1 = nn.Conv1d(4, 32, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv1d(32, self.hidden_size, kernel_size=3, stride=2)

        self.dense_out = nn.Linear(self.hidden_size*3, 4)

        with open(label_stats, 'r') as f: 
            lines = f.readlines()
        self.M = torch.FloatTensor([float(_b) for _b in lines[1].split(',')]).cuda()
        self.S = torch.FloatTensor([float(_b) for _b in lines[2].split(',')]).cuda()


    def forward(self, x):
        x = x[:,:,1:] # not use exact time feature

        # ==== dynamic features ====
        out = self.dense_1( x )
        _, out = self.rnn(out) # tuple (hidden, cell)
        dynamic_out = out[0][-1] # bsz, dim

        # ==== static features ====
        _min, _ = torch.min(x, 1) # bsz,4
        _max, _ = torch.max(x, 1)
        _mean = torch.mean(x, 1)
        _std = torch.std(x, 1)
        
        static_feat = torch.cat(
            [_min, _max, _mean, _std], -1
        )

        out = F.relu(self.dense_2(static_feat))
        static_out = F.relu(self.dense_3(out))


        # ==== conv features ====
        conv_in = x.transpose(-1,-2) # bsz,4,t
        out = F.relu(self.conv_1(conv_in))
        out = F.relu(self.conv_2(out))
        conv_out = out.mean(-1) # average pooling

        feat = torch.cat([dynamic_out, static_out, conv_out], dim=-1) 
        out = self.dense_out(feat)
        out = out * self.S.unsqueeze(0) + self.M.unsqueeze(0)
        return out
