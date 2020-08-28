import numpy as np
import scipy

def moving_average(data, l=30):
    out = []
    if type(data) == list:
        data = np.array(data)
    for i in range(l, data.shape[0]):
        ma = np.mean(data[i-l:i])
        out.append(ma)
    out = np.array(out)
    return out


def lqr(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R).dot(B.T).dot(P)
    return -K


def dlqr(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(np.dot(B.T, P).dot(B) + R).dot(B.T).dot(P).dot(A)
    return -K


def discretized_system(A, B, dt):
    Ad = np.eye(A.shape[0]) + dt * A
    Bd = dt * B
    return Ad, Bd
