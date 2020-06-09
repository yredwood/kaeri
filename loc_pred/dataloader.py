import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb

class CustomLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.data)

#    def __getitem__(self, index):
#        key, x, y, m, v, dist, h, l = self.data[index]
#        
#        if x is not None:
#            label = np.stack([x,y,m,v])
#        else:
#            label = np.zeros([0])
#        data = np.concatenate([dist, h, l], -1)
#
#        return data, label, key

    def __getitem__(self, index):
        d = self.data[self.keys[index]]
        data = d['static']
        if 'label' in d.keys():
            label = d['label']
        else:
            label = np.zeros([0])
        return data, label, self.keys[index]

    def get_full_data(self):
        d_all, l_all, i_all = [], [], []
        for i in range(len(self)):
            data, label, _id = self[i]
            d_all.append(data)
            l_all.append(label)
            i_all.append(_id)

        return np.array(d_all), np.array(l_all), i_all


class Collater():

    def __init__(self):
        pass

    def __call__(self, batch):
        data = torch.FloatTensor([b[0] for b in batch])
        label = torch.FloatTensor([b[1] for b in batch])
        key = [b[2] for b in batch]
        return data, label, key

if __name__ == '__main__':

    out = []
    label = [] 
    def get_data(path, _l):
        _set = CustomLoader(path)
        for i in range(len(_set)):
            out.append(_set[i][0])
            label.append(_l)

    get_data('data/train.pkl', 0)
    get_data('data/val.pkl', 1)
    get_data('data/test.pkl', 2)

    pdb.set_trace()

    tsne_emb = TSNE(n_components=2).fit_transform(out)
    d0 = tsne_emb[np.array(label) == 0]
    d1 = tsne_emb[np.array(label) == 1]
    d2 = tsne_emb[np.array(label) == 2]

    plt.plot(d0[:,0], d0[:,1], 'r.')
    plt.plot(d1[:,0], d1[:,1], 'g.')
    plt.plot(d2[:,0], d2[:,1], 'b.')

    plt.savefig('datadist.png')
