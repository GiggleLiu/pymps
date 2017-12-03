import numpy as np
import re

re_float = r"[-+]?\d*\.\d+|\d+"


def format_factor(num):
    if abs(num - 1) < 1e-5:
        return ''
    if abs(np.imag(num)) < 1e-5:
        num = np.real(num)
    if np.imag(num) == 0 and abs(np.fmod(np.real(num), 1)) < 1e-5:
        num = int(num)
    res = '%s*' % np.around(num, decimals=3)
    return res
