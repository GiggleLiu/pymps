from matplotlib.pyplot import *

__all__ = ['show_bm']


def show_bm(bm, tokens=None, ax=None, **kwargs):
    '''
    Display a mesh of block indexer.

    Args:
        bm (<BlockMarker>): the block marker.
        tokens (list): show tokens in these blocks.
        ax (<axis>): the target axis.
        :key word arguments: passed to matplotlib.plot method.
    '''
    Nr = bm.Nr
    ndim = Nr[-1]
    if ax is None:
        ax = gca()
    if tokens is None:
        tokens = bm.qns
    for i in range(bm.nblock):
        start, stop = Nr[i], Nr[i + 1]
        width = stop - start
        if width == 0:
            continue
        rect = Rectangle((start, start), width, width,
                         edgecolor='r', fill=False)
        ax.add_patch(rect)
        text(start + width / 2., start + width / 2.,
             tokens[i], verticalalignment='center', horizontalalignment='center', fontsize=10)
        ax.plot([stop, stop], [0, ndim], ls=':', color='k', **kwargs)
        ax.plot([0, ndim], [stop, stop], ls=':', color='k', **kwargs)
    ax.plot([0, ndim], [0, 0], ls=':', **kwargs)
    ax.plot([0, 0], [0, ndim], ls=':', **kwargs)
    ymin, ymax = ax.get_ylim()
    if ymin < ymax:
        ax.invert_yaxis()
