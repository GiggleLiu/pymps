import numpy as np
from plotlib import plt, matplotlib

import pdb
import time


def cornerlabel(txt):
    '''add corner label.'''
    text(0.02, 0.9, txt, transform=gca().transAxes, fontsize=14)


def halfc(xy, xscale, yscale, **kwargs):
    '''draw half circle'''
    theta = linspace(0, pi)
    x = sin(theta) * xscale + xy[0]
    y = cos(theta) * yscale + xy[1]
    plot(x, y, **kwargs)


class MPSPloter(object):
    def __init__(self, **kwargs):
        pass

    def ket(self, offset=(0, 0), L=6, withflow=False, labelmask=[True] * 3, hatch='', index_offset=0):
        x, y = offset
        flowmask = [0, -1, 1] if withflow else [0] * 3
        for i in xrange(L):
            hatchi = hatch if not hasattr(hatch, '__iter__') else hatch[i]
            legmask = ones(3)
            if i == 0:
                legmask[0] = 0
            if i == L - 1:
                legmask[-1] = 0
            lbs = [r'$a_%s$' % (i + index_offset), '$\sigma_%s$' %
                   (i + 1 + index_offset), '']
            draw_kb(x, y, labels=[l if m else '' for m, l in zip(labelmask, lbs)], legmask=legmask,
                    flowmask=flowmask, hatch=hatchi, facecolor='k' if hatchi == '' else 'w')
            if i != L - 1:
                x += 1
        axis('equal')
        axis('off')

    def bra(self, offset=(0, 0), L=6, withflow=False, labelmask=[True] * 3, hatch='', index_offset=0):
        x, y = offset
        flowmask = [0, 1, -1] if withflow else [0] * 3
        for i in xrange(L):
            hatchi = hatch if not hasattr(hatch, '__iter__') else hatch[i]
            legmask = ones(3)
            if i == 0:
                legmask[0] = 0
            if i == L - 1:
                legmask[-1] = 0
            lbs = [r"$a'_%s$" % (i + index_offset),
                   r"$\sigma_%s'$" % (i + index_offset + 1), '']
            draw_kb(x, y, labels=[l if m else '' for m, l in zip(labelmask, lbs)], legmask=legmask,
                    flowmask=flowmask, hatch=hatchi, is_ket=False, facecolor='k' if hatchi == '' else 'w')
            if i != L - 1:
                x += 1
        axis('equal')
        axis('off')

    def mpo(self, offset=(0, 0), L=6, withflow=False, labelmask=[True] * 4, texts=None):
        x, y = offset
        flowmask = [0, 1, -1, -1] if withflow else [0] * 4
        for i in xrange(L):
            legmask = ones(4)
            if i == 0:
                legmask[0] = 0
            if i == L - 1:
                legmask[-1] = 0
            lbs = [r'$b_%s$' % i, r"$\sigma_%s'$" %
                   (i + 1), r"$\sigma_%s$" % (i + 1), '']
            draw_sq(x, y, labels=[l if m else '' for m, l in zip(labelmask, lbs)], legmask=legmask,
                    flowmask=flowmask, zoom=1., facetext=('', 'r', 16) if texts is None else texts[i])
            if i != L - 1:
                x += 1
        axis('equal')
        axis('off')

    def box(self, start, end, offset=(0, 0), color='b'):
        '''Box mps from start to end.'''
        ax = gca()
        x = offset[0] - 0.5 + start
        y = offset[1] - 0.5
        r1 = Rectangle((x, y), end - start + 1, 1,
                       facecolor='none', ls='--', lw=2, edgecolor=color)
        ax.add_patch(r1)
