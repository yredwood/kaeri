import numpy as np
import pdb


num_ensemble = 32
epoch = 400
#output_fname = 'finals/relu_bigger_2layer_{}_{:.3f}.csv'
output_fname = 'finals/jiyu4_2layer_{}_{:.3f}.csv'
fname_list = ['submissions/jiyu4_2layer_{}_epoch{}.csv'.format(i, epoch) for i in range(num_ensemble)]
#fname_list = ['submissions/jiyu3_{}_epoch{}.csv'.format(i, epoch) for i in range(4,15)]
#fname_list = ['finals/ensemble_0610_2.csv', 'finals/relu_bigger_layer_0.019.csv']

def get_lines(fname):
    with open(fname, 'rt') as f: 
        lines = f.readlines()
    return lines

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = 1. / np.array(x) / 10.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


_lines = [get_lines(fname) for fname in fname_list]

new_lines = [_lines[0][0].strip()]
losses = [float(line[-1].strip()) for line in _lines]

w = softmax(losses)
for i in range(1, len(_lines[0])-1):
    # last line is validation loss
    current_lines = [line[i].strip().split(',') for line in _lines]

    _id = current_lines[0][0]
    cline = '{}'.format(_id)
    for j in range(1,5):
        _k = np.mean([float(crl[j]) for crl in current_lines])
        #_k = np.sum([float(current_lines[_i][j]) * w[_i] for _i in range(len(current_lines))])
        cline += ',{:.4f}'.format(_k)
    
    new_lines.append(cline)

# get avg loss
output_fname = output_fname.format(epoch, np.mean(losses))

with open(output_fname, 'wt') as f:
    f.writelines('\n'.join(new_lines))
    print (output_fname)
