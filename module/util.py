import numpy as np

def moving_average(data, l=30):
    out = []
    if type(data) == list:
        data = np.array(data)
    for i in range(l, data.shape[0]):
        ma = np.mean(data[i-l:i])
        out.append(ma)
    out = np.array(out)
    return out
