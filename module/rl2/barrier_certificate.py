import numpy as np

def u_of_x(x, u_min, u_max, k=4):
    assert x.shape[0]==2, 'shape_error'
    assert u_min < u_max, f'input constraint error. {u_min} > {u_max}?'
    m = 1
    l = 1
    g = 10.
    dt = .05
    gamma = pow(dt, -k)
    th, thd = x
    if th != 0:
        u_thres = np.roots([-(6*th*dt**2)/(m*l**2), -(2*th**2 + 2*th*thd*dt + 3*g/l*th*np.sin(th)*dt**2 + gamma*(th**2-1))])[0]
        if th < 0:
            u_thres = max(u_thres, u_min)
            ran = [u_thres, u_max]
        else:
            u_thres = min(u_thres, u_max)
            ran = [u_min, u_thres]
    return ran

def u_cbf(x, u_candidate, u_min, u_max):
    ran = u_of_x(x, u_min, u_max)
    rmin, rmax = ran[0], ran[1]
    out = u_candidate
    if u_candidate < u_min:
        out = u_min
    elif u_candidate > u_max:
        out = u_max
    return np.array([out])
