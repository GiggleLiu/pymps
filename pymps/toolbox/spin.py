import numpy as np

__all__ = ['sx', 'sy', 'sz', 's', 's1x', 's1y', 's1z', 's1', 's2vec', 'vec2s']

# pauli spin
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
s = [np.identity(2), sx, sy, sz]

# spin 1 matrices.
s1x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
s1y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
s1z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
s1 = [np.identity(3), s1x, s1y, s1z]


def s2vec(s):
    '''
    Transform a spin to a 4 dimensional vector, corresponding to s0,sx,sy,sz component.

    s: 
        the spin.
    '''
    res = np.array([np.trace(s), np.trace(np.dot(sx, s)), np.trace(
        np.dot(sy, s)), np.trace(np.dot(sz, s))]) / 2
    return res


def vec2s(n):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    n: 
        a 1-D array of length 3 or 4 to specify the `direction` of spin.
    *return*:
        2 x 2 matrix.
    '''
    if len(n) <= 3:
        res = np.zeros([2, 2], dtype='complex128')
        for i in range(len(n)):
            res += s[i + 1] * n[i]
        return res
    elif len(n) == 4:
        return np.identity(2) * n[0] + sx * n[1] + sy * n[2] + sz * n[3]
    else:
        raise Exception('length of vector %s too large.' % len(n))
