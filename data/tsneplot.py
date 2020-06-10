from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

import pickle
import pdb

with open('statistics.pkl', 'rb') as f: 
    data =pickle.load(f)


train_data, test_data = data

full_data = np.concatenate(data, 0)
labels = np.ones(len(full_data))
labels[:len(train_data)] = 0


tsne_emb = TSNE(n_components=2).fit_transform(full_data)

d0 = tsne_emb[labels==0]
d1 = tsne_emb[labels==1]

plt.plot(d0[:,0], d0[:,1], 'r.')
plt.plot(d1[:,0], d1[:,1], 'b+')
#plt.plot(d0, np.zeros(len(d0)), 'r.')
#plt.plot(d1, np.zeros(len(d1))+1, 'b+')
plt.savefig('dist.png')
