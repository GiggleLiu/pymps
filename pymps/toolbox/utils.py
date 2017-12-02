from functools import wraps
import pickle

__all__ = ['inherit_docstring_from', 'quicksave', 'quickload', 'nullobj']

def inherit_docstring_from(cls):
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return docstring_inheriting_decorator


def quicksave(filename, obj):
    '''
    save an instance.

    Args:
        filename (str): file to save this object.
        obj (object): the objcet to save.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def quickload(filename):
    '''
    load an instance.

    Args:
        filename (str): to filename to save this object.

    Returns:
        obj (object): loaded objcet.
    '''
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def nullobj():
    '''an empty object'''
    return type('obj', (object,), {})()
