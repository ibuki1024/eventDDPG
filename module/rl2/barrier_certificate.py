import numpy as np


def u_cbf(x, u_candidate, ratio):
    ran = _u_of_x(x, ratio)
    rmin, rmax = ran[0], ran[1]
    out = u_candidate
    if u_candidate < rmin:
        out = rmin
    elif u_candidate > rmax:
        out = rmax
    return np.array([out])


def _u_of_x(x, ratio, k=4):
    assert x.shape[0]==2, 'shape_error'
    m = 1
    l = 1
    g = 10.
    dt = .05
    gamma = pow(dt, -k)
    th, thd = x
    if th != 0:
        u_thres = np.roots([-(6*th*dt**2)/(m*l**2), -(2*th**2 + 2*th*thd*dt + 3*g/l*th*np.sin(th)*dt**2 + gamma*(th**2-1))])[0]
        if th < 0:
            u_thres = max(u_thres, -ratio)
            ran = [u_thres, ratio]
        else:
            u_thres = min(u_thres, ratio)
            ran = [-ratio, u_thres]
    return ran
