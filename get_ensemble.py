import numpy as np
import pdb


num_ensemble = 10
output_fname = 'finals/nograd_{:.3f}.csv'
fname_list = ['submissions/grad_or_nograd_{}_epoch400.csv'.format(i) for i in range(num_ensemble)]

def get_lines(fname):
    with open(fname, 'rt') as f: 
        lines = f.readlines()
    return lines

_lines = [get_lines(fname) for fname in fname_list]

new_lines = [_lines[0][0].strip()]
for i in range(1, len(_lines[0])-1):
    # last line is validation loss
    current_lines = [line[i].strip().split(',') for line in _lines]

    _id = current_lines[0][0]
    cline = '{}'.format(_id)
    for j in range(1,5):
        _k = np.mean([float(crl[j]) for crl in current_lines])
        cline += ',{:.4f}'.format(_k)
    
    new_lines.append(cline)

# get avg loss
losses = [float(line[-1].strip()) for line in _lines]
output_fname = output_fname.format(np.mean(losses))

with open(output_fname, 'wt') as f:
    f.writelines('\n'.join(new_lines))
