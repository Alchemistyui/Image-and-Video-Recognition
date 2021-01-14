import torch
import numpy as np
import json

import pdb

ewc = {'wspc': {}, 'swpc': {}, 'spwc': {}, 'spcw': {}}
no_ewc = {'wspc': {}, 'swpc': {}, 'spwc': {}, 'spcw': {}}

def avg(d):
    out = {}
    for i in d.keys():
        if d[i] != {}:
            mtx = torch.zeros([5, 4], dtype=torch.float64)
            acc = []
            bwd = []
            fwd = []
            # pdb.set_trace()
            for j in d[i]:
                mtx += d[i][j][0]
                acc.append(d[i][j][1])
                bwd.append(d[i][j][2])
                fwd.append(d[i][j][3])

            pdb.set_trace()

            out[i] = [(mtx/5).numpy().tolist(), 
                [np.mean(acc), min(max(acc)-np.mean(acc), np.mean(acc)-min(acc))],
                [np.mean(bwd), min(max(bwd)-np.mean(bwd), np.mean(bwd)-min(bwd))],
                [np.mean(fwd), min(max(fwd)-np.mean(fwd), np.mean(fwd)-min(fwd))]]

    return out


ewc['wspc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.6667, 0.4050, 0.5000, 0.4792],
        [0.8333, 0.9750, 0.9444, 0.7292],
        [0.8472, 0.9750, 1.0000, 0.8542],
        [0.9583, 0.9250, 1.0000, 0.9792]], dtype=torch.float64),
        0.965625, 0.080556, 0.734537]

ewc['wspc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7083, 0.4300, 0.4167, 0.4792],
        [0.7917, 0.9750, 0.9444, 0.7083],
        [0.8472, 0.9500, 1.0000, 0.8750],
        [0.9167, 0.9500, 1.0000, 0.9792]], dtype=torch.float64), 
        0.961458, 0.061111, 0.749815]

ewc['wspc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7083, 0.4700, 0.4722, 0.4792],
        [0.7917, 0.9750, 0.9444, 0.7083],
        [0.7361, 0.9500, 1.0000, 0.8750],
        [0.9167, 0.9250, 1.0000, 0.9792]], dtype=torch.float64),
        0.955208, 0.052778, 0.763148]

ewc['wspc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7500, 0.4550, 0.5139, 0.4583],
        [0.8333, 0.9500, 0.9444, 0.7292],
        [0.6944, 0.9750, 1.0000, 0.8958],
        [0.9167, 0.9500, 1.0000, 0.9792]], dtype=torch.float64),
        0.961458, 0.055556, 0.765093]

ewc['wspc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7500, 0.4050, 0.5000, 0.3542],
        [0.7083, 0.9500, 0.9444, 0.7500],
        [0.8056, 0.9500, 1.0000, 0.9167],
        [0.9167, 0.9250, 1.0000, 0.9792]], dtype=torch.float64),
        0.955208, 0.047222, 0.755370]


ewc['swpc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9750, 0.1667, 0.2222, 0.2500],
        [0.8500, 0.9167, 0.9167, 0.8333],
        [0.9500, 0.7778, 1.0000, 0.9375],
        [0.9100, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.934097, -0.044815, 0.673611]


ewc['swpc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.1667, 0.3611, 0.3750],
        [0.9250, 0.9167, 0.9583, 0.8333],
        [0.9500, 0.7361, 1.0000, 0.9375],
        [0.9500, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.944097, -0.023148, 0.687500]

ewc['swpc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9000, 0.2083, 0.2083, 0.2500],
        [0.8500, 0.8750, 0.9167, 0.8542],
        [0.9500, 0.7361, 1.0000, 0.8958],
        [0.9100, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.934097, -0.005926, 0.673611]


ewc['swpc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9250, 0.1667, 0.2083, 0.2708],
        [0.8750, 0.8750, 0.9167, 0.8125],
        [0.9500, 0.8889, 1.0000, 0.9583],
        [0.9100, 0.9583, 1.0000, 0.9792]], dtype=torch.float64),
        0.961875, 0.022778, 0.680556]

ewc['swpc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.1667, 0.2222, 0.2708],
        [0.8250, 0.9583, 0.9028, 0.8125],
        [0.9000, 0.7778, 0.9444, 0.9167],
        [0.9750, 0.8472, 1.0000, 0.9583]], dtype=torch.float64),
        0.945139, -0.010185, 0.662037]


ewc['spwc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9250, 0.3056, 0.2083, 0.3542],
        [0.9500, 1.0000, 0.5278, 0.7917],
        [0.8500, 1.0000, 0.8889, 0.8958],
        [0.9350, 1.0000, 0.9583, 0.9792]], dtype=torch.float64),
        0.968125, 0.026481, 0.576389]

ewc['spwc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9750, 0.2222, 0.1667, 0.3333],
        [0.9750, 1.0000, 0.4167, 0.6458],
        [0.9000, 1.0000, 0.8889, 0.9167],
        [0.9350, 1.0000, 0.9167, 0.9792]], dtype=torch.float64),
        0.957708, -0.004074, 0.518519]

ewc['spwc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9000, 0.3889, 0.1250, 0.2292],
        [0.9750, 1.0000, 0.5000, 0.7292],
        [0.9000, 1.0000, 0.9583, 0.8958],
        [0.9350, 1.0000, 0.8472, 0.9792]], dtype=torch.float64),
        0.940347, -0.025370, 0.594907]

ewc['spwc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.4028, 0.1250, 0.3125],
        [0.9750, 1.0000, 0.4167, 0.7917],
        [0.9250, 1.0000, 1.0000, 0.9167],
        [0.9350, 1.0000, 0.9583, 0.9792]], dtype=torch.float64),
        0.968125, -0.018889, 0.578704]

ewc['spwc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.2639, 0.1667, 0.2708],
        [0.9500, 1.0000, 0.5417, 0.7500],
        [0.9250, 1.0000, 1.0000, 0.8958],
        [0.9350, 1.0000, 0.8472, 0.9792]], dtype=torch.float64),
        0.940347, -0.055926, 0.567130]

ewc['spcw']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9250, 0.2778, 0.3750, 0.1667],
        [0.9500, 1.0000, 0.7708, 0.4583],
        [0.8850, 1.0000, 0.9792, 0.5694],
        [0.9000, 1.0000, 0.9792, 1.0000]], dtype=torch.float64),
        0.969792, -0.008333, 0.539352]

ewc['spcw']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.4722, 0.2708, 0.2083],
        [0.9500, 1.0000, 0.7500, 0.4583],
        [0.9500, 1.0000, 0.9792, 0.5694],
        [0.9500, 1.0000, 1.0000, 1.0000]], dtype=torch.float64),
        0.987500, 0.020278, 0.597222]

ewc['spcw']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.8850, 0.2778, 0.2917, 0.3194],
        [0.9500, 1.0000, 0.7708, 0.5000],
        [0.9750, 1.0000, 0.9792, 0.5694],
        [0.9250, 1.0000, 1.0000, 0.9583]], dtype=torch.float64),
        0.970833, 0.020278, 0.539352]

ewc['spcw']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.2361, 0.3125, 0.1667],
        [0.9500, 1.0000, 0.7292, 0.5000],
        [0.9350, 1.0000, 0.9792, 0.5694],
        [0.9250, 1.0000, 1.0000, 0.9583]], dtype=torch.float64),
        0.970833, 0.011944, 0.511574]

ewc['spcw']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9750, 0.3333, 0.2292, 0.0833],
        [0.9500, 1.0000, 0.7917, 0.3750],
        [0.9750, 1.0000, 0.9792, 0.5694],
        [0.9250, 1.0000, 0.9792, 0.8889]], dtype=torch.float64),
        0.948264, -0.016667, 0.564815]







no_ewc['wspc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7500, 0.4950, 0.5000, 0.3542],
        [0.7500, 0.9750, 0.9444, 0.7083],
        [0.8056, 0.9500, 1.0000, 0.8750],
        [0.9583, 0.9750, 1.0000, 0.9792]], dtype=torch.float64),
        0.978125, 0.069444, 0.771481]

no_ewc['wspc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7083, 0.4300, 0.4167, 0.4583],
        [0.7500, 0.9500, 0.9583, 0.7917],
        [0.8056, 0.9250, 1.0000, 0.9375],
        [0.9167, 0.9250, 1.0000, 0.9792]], dtype=torch.float64), 
        0.955208, 0.061111, 0.775278]

no_ewc['wspc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.6667, 0.4200, 0.5139, 0.4792],
        [0.7917, 0.9250, 0.9444, 0.7292],
        [0.8472, 0.9250, 1.0000, 0.8750],
        [0.9167, 0.8600, 0.9583, 0.9792]], dtype=torch.float64),
        0.928542, 0.047778, 0.746481]

no_ewc['wspc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.6667, 0.4300, 0.5417, 0.3958],
        [0.8333, 0.9750, 1.0000, 0.7500],
        [0.8472, 0.9750, 1.0000, 0.8750],
        [0.9583, 0.9250, 1.0000, 0.9792]], dtype=torch.float64),
        0.965625, 0.080556, 0.768333]

no_ewc['wspc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.7083, 0.4150, 0.5833, 0.3958],
        [0.7917, 0.9750, 1.0000, 0.7292],
        [0.7361, 0.9750, 1.0000, 0.8750],
        [0.9167, 0.9250, 1.0000, 0.9792]], dtype=torch.float64),
        0.955208, 0.052778, 0.763333]

no_ewc['swpc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.2083, 0.4028, 0.3125],
        [0.8750, 0.9167, 0.9583, 0.8333],
        [0.9750, 0.8889, 1.0000, 0.9167],
        [0.9350, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.940347, -0.014815, 0.694444]

no_ewc['swpc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.1667, 0.2778, 0.3750],
        [0.8500, 0.8056, 0.9583, 0.8542],
        [0.9500, 0.8889, 1.0000, 0.9375],
        [0.9600, 0.7361, 1.0000, 0.9792]], dtype=torch.float64),
        0.918819, -0.006481, 0.687500]

no_ewc['swpc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.1667, 0.1667, 0.2500],
        [0.8000, 0.7639, 0.9167, 0.7917],
        [0.9750, 0.7778, 1.0000, 0.9583],
        [0.9600, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.946597, 0.044444, 0.680556]

no_ewc['swpc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.8850, 0.1667, 0.3889, 0.3333],
        [0.8250, 0.8472, 0.9167, 0.8333],
        [0.9750, 0.7778, 1.0000, 0.9375],
        [0.9500, 0.8472, 1.0000, 0.9792]], dtype=torch.float64),
        0.944097, 0.021667, 0.673611]

no_ewc['swpc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.8850, 0.1667, 0.2778, 0.4167],
        [0.8750, 0.7222, 0.9167, 0.7917],
        [0.9750, 0.8889, 1.0000, 0.9583],
        [0.9750, 0.8472, 1.0000, 0.9583]], dtype=torch.float64),
        0.945139, 0.071667, 0.680556]


no_ewc['spwc']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9350, 0.3472, 0.1250, 0.2917],
        [0.9750, 1.0000, 0.5417, 0.7708],
        [0.9500, 1.0000, 1.0000, 0.8542],
        [0.9100, 1.0000, 0.9583, 0.9792]], dtype=torch.float64),
        0.961875, -0.022222, 0.581019]

no_ewc['spwc']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.2639, 0.1667, 0.2500],
        [0.9750, 1.0000, 0.4583, 0.7292],
        [0.8750, 1.0000, 1.0000, 0.9167],
        [0.9350, 1.0000, 0.8472, 0.9792]], dtype=torch.float64),
        0.940347, -0.055926, 0.546296]

no_ewc['spwc']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9000, 0.3472, 0.1667, 0.2500],
        [0.9500, 1.0000, 0.4583, 0.7292],
        [0.9250, 1.0000, 0.9167, 0.9375],
        [0.9350, 1.0000, 0.8472, 0.9792]], dtype=torch.float64),
        0.940347, -0.011481, 0.581019]

no_ewc['spwc']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.3611, 0.1667, 0.3125],
        [0.9750, 1.0000, 0.4583, 0.7917],
        [0.9500, 1.0000, 0.8889, 0.8958],
        [0.9750, 1.0000, 0.8472, 0.9792]], dtype=torch.float64),
        0.950347, 0.007778, 0.571759]

no_ewc['spwc']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.3889, 0.2778, 0.2500],
        [0.9750, 1.0000, 0.4167, 0.7292],
        [0.9000, 1.0000, 0.8472, 0.9375],
        [0.9500, 1.0000, 0.9167, 0.9792]], dtype=torch.float64),
        0.961458, 0.036481, 0.581019]


no_ewc['spcw']['exp1'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.8600, 0.1250, 0.2708, 0.1667],
        [0.9750, 1.0000, 0.7292, 0.4167],
        [0.9350, 1.0000, 0.9792, 0.5694],
        [0.9250, 1.0000, 0.9792, 0.9583]], dtype=torch.float64),
        0.965625, 0.021667, 0.474537]

no_ewc['spcw']['exp2'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9350, 0.2500, 0.2500, 0.1250],
        [0.9500, 1.0000, 0.7500, 0.4583],
        [0.9750, 1.0000, 0.9792, 0.6528],
        [0.9250, 1.0000, 0.9792, 0.9583]], dtype=torch.float64),
        0.965625, -0.003333, 0.550926]

no_ewc['spcw']['exp3'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9350, 0.2222, 0.2500, 0.1667],
        [0.9500, 1.0000, 0.7708, 0.5000],
        [0.9500, 1.0000, 0.9792, 0.5694],
        [0.9000, 1.0000, 1.0000, 0.8889]], dtype=torch.float64),
        0.947222, -0.004722, 0.520833]

no_ewc['spcw']['exp4'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9350, 0.3056, 0.2917, 0.2083],
        [0.9750, 1.0000, 0.7708, 0.5000],
        [0.9750, 1.0000, 0.9792, 0.5694],
        [0.9750, 1.0000, 1.0000, 0.9583]], dtype=torch.float64),
        0.983333, 0.020278, 0.548611]

no_ewc['spcw']['exp5'] = [torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.9500, 0.3194, 0.2917, 0.1667],
        [0.9500, 1.0000, 0.7708, 0.4583],
        [0.9350, 1.0000, 0.9792, 0.6111],
        [0.9100, 1.0000, 0.9792, 0.8889]], dtype=torch.float64),
        0.944514, -0.013333, 0.567130]





print(avg(ewc))

# print(avg(no_ewc))

# with open('./out_avg_results.json', 'w') as f:
#     f.write(json.dumps(avg(ewc)))
#     f.write('\n')

# with open('./out_avg_results.json', 'a') as f:
#     f.write(json.dumps(avg(no_ewc)))




