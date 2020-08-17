import numpy as np


def u_cbf(x, u_candidate, ratio):
    ran = _u_of_x(x, ratio)
    rmin, rmax = ran[0], ran[1]
    out = u_candidate
    if h(x) < 1e-3:
        if u_candidate < rmin:
            out = rmax
        elif u_candidate > rmax:
            out = rmin
    else:
        if u_candidate < rmin:
            out = rmin
        elif u_candidate > rmax:
            out = rmax
    return np.array([out])


def h(x, alpha=0.4):
        return 1 - x[0]**2 - alpha*x[1]**2


def _u_of_x(x, ratio, k=2.6, alpha=0.4):
    assert x.shape[0]==2, 'shape_error'
    al = alpha
    m = 1
    l = 1
    g = 10.
    gamma = pow(10, k)
    th, thd = x
    if thd == 0:
        p = abs(m*g*np.sin(th)/2)
        ran = [-ratio, -p] if th > 0 else [p, ratio]
    else:
        u_thres = np.roots([-(6*al*thd)/(m*l**2), -(2*th*thd + 3*al*g/l*thd*np.sin(th) - gamma*h(x, al))])[0]
        if thd < 0:
            u_thres = max(u_thres, -ratio)
            ran = [u_thres, ratio]
        else:
            u_thres = min(u_thres, ratio)
            ran = [-ratio, u_thres]
    return ran
