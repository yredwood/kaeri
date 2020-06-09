import numpy as np
import pdb


num_ensemble = 12
output_fname = 'xgb_ensemble.csv'
fname_list = ['xgb_{}.csv'.format(i) for i in range(num_ensemble)]


def get_lines(fname):
    with open(fname, 'rt') as f: 
        lines = f.readlines()
    return lines

_lines = [get_lines(fname) for fname in fname_list]

new_lines = [_lines[0][0].strip()]
for i in range(1, len(_lines[0])):

    current_lines = [line[i].strip().split(',') for line in _lines]

    _id = current_lines[0][0]
    cline = '{}'.format(_id)
    for j in range(1,5):
        _k = np.mean([float(crl[j]) for crl in current_lines])
        cline += ',{:.4f}'.format(_k)
    
    new_lines.append(cline)

with open(output_fname, 'wt') as f:
    f.writelines('\n'.join(new_lines))
