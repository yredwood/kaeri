import torch
import pickle
from torch.utils.data import DataLoader
import pdb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# data loader
class CustomLoader(torch.utils.data.Dataset):
    def __init__(self, path):

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        self.ids = list(self.data.keys())
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        output = {}
        selected_data = self.data[self.ids[index]]

        output['data'] = selected_data['data']
        if 'label' in selected_data.keys():
            output['label'] = selected_data['label']

        if 'static' in selected_data.keys():
            output['static'] = selected_data['static']
    
        output['oid'] = self.ids[index]

        return output

class Collater():

    def __init__(self):
        pass

    def __call__(self, batch):
        # batch = [b['data', 'label]..   ]

        data_length = [len(b['data']) for b in batch]
        data = torch.FloatTensor([b['data'] for b in batch])

        if 'label' in batch[0].keys():
            label = torch.FloatTensor([b['label'] for b in batch])
        else:
            label = None

        oids = [b['oid'] for b in batch]
        
        return data, label, oids



if __name__ == '__main__':

    _train = CustomLoader('data/train.pkl')
    _test = CustomLoader('data/test.pkl')
    _val = CustomLoader('data/val.pkl')

    collate_fn = Collater()

#    train_loader = DataLoader(customloader, num_workers=1, shuffle=False,
#            batch_size=12, pin_memory=False, drop_last=True, collate_fn=collate_fn) 
#
#    test_loader = DataLoader(customloader, num_workers=1, shuffle=False,
#            batch_size=12, pin_memory=False, drop_last=True, collate_fn=collate_fn) 
#
#    val_loader = DataLoader(customloader, num_workers=1, shuffle=False,
#            batch_size=12, pin_memory=False, drop_last=True, collate_fn=collate_fn) 
#
#    for i, (data, label, _) in enumerate(train_loader):


    # visualise
    
    out = []
    label = []
    def get_data(path, _l):
        _set = CustomLoader(path)

        for i in range(len(_set)):
            out.append(_set[i]['static'])
            label.append(_l)

    get_data('data/train.pkl', 0)
    get_data('data/test.pkl', 1)
    get_data('data/val.pkl', 2)

    tsne_embedded = TSNE(n_components=1).fit_transform(out)
    
    d0 = tsne_embedded[np.array(label) == 0]
    d1 = tsne_embedded[np.array(label) == 1]
    d2 = tsne_embedded[np.array(label) == 2]
    
#    plt.plot(d0[:,0], d0[:,1], 'r.')
#    plt.plot(d1[:,0], d1[:,1], 'b.')
#    plt.plot(d2[:,0], d2[:,1], 'g.')
    plt.plot(d0, np.zeros(len(d0)), 'r+')
    plt.plot(d1, np.zeros(len(d1))+1, 'b*')
    plt.plot(d2, np.zeros(len(d2))+2, 'g.')
    
    
    plt.savefig('_datadist.png')

    pdb.set_trace()


    








    



#
