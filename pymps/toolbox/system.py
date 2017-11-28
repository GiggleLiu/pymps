import os
from sys import platform

__all__ = ['openfile']


def openfile(filename):
    '''
    Open a file.

    Args:
        filename (str): the target file.

    Return:
        bool: succeed if True.
    '''
    if platform == "linux" or platform == "linux2":
        os.system('xdg-open %s' % filename)
    elif platform == "darwin":
        os.system('open %s' % filename)
    elif platform == "win32":
        os.startfile(filename)
    else:
        print('Can not open file, platform %s not handled!' % platform)
        return False
    return True
