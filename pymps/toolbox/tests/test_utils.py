from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
import pdb

from ..utils import *


class A(object):
    def f1(self, x):
        '''x^2'''
        return x**2


class B(A):
    @inherit_docstring_from(A)
    def f1(self, x):
        return x**3


def test_docstring_saveload():
    print('Test inherit docstring.')
    b = B()
    assert_(b.f1.__doc__ == 'x^2')
    assert_(b.f1(3) == 27)

    print('Test Save and load')
    filename = 'savetest.dat'
    quicksave(filename, B())
    c = quickload(filename)
    assert_(c.f1.__doc__ == 'x^2')
    assert_(c.f1(3) == 27)


if __name__ == '__main__':
    test_docstring_saveload()
