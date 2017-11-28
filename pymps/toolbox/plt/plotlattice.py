import numpy as np
from .plotlib import plt, matplotlib
from matplotlib import mlab, cm, colors

SITE_THEME_DICT = {
    'basic': ('k', 'uniform'),
    'black-fancy': ('k', 'star'),
    'black': ('k', 'uniform'),
}
'''edgecolor, style'''


def cornerlabel(txt):
    '''add corner label.'''
    text(0.02, 0.9, txt, transform=gca().transAxes)


def rotate_leg(vec, theta):
    '''Rotate leg.'''
    mat = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return mat.dot(transpose(vec)).T


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return colors.LinearSegmentedColormap('CustomMap', cdict)


class LatticePloter(object):
    def __init__(self, radius=0.2, lw=2, color='#333333'):
        self.radius = radius
        self._cm = cm.gray
        self.lw = lw
        self.color = color

    def __setattr__(self, name, val):
        '''set color of sites.'''
        self.__dict__[name] = val
        if name == 'color':
            c = colors.ColorConverter().to_rgb
            if isinstance(val, str):
                val = c(val)
            self._cm = make_colormap([val, c('white')])

    def plot_site(self, ax, xy, kind='basic'):
        '''show two sites'''
        x, y = xy
        radius = self.radius

        edgecolor, style = SITE_THEME_DICT[kind]
        patch = plt.Circle(xy, radius, facecolor=self.color, edgecolor='none')
        ax.add_patch(patch)

        if style == 'star':
            xl = np.linspace(x - radius * 1.1, x + radius * 1.1, 200)
            yl = np.linspace(y - radius * 1.1, y + radius * 1.1, 200)
            X, Y = np.meshgrid(xl, yl)
            Z = mlab.bivariate_normal(
                X, Y, sigmax=radius * 0.6, sigmay=radius * 0.6, mux=x + radius / 4., muy=y + radius / 4.)

            im = plt.imshow(Z, interpolation='bilinear', cmap=self._cm,
                            origin='lower', extent=[x - radius, x + radius, y - radius, y + radius],
                            clip_path=patch, clip_on=True, zorder=100)
            im.set_clip_path(patch)

    def connect_sites(self, xy1, xy2):
        '''connect two sites'''
        xy1, xy2 = asarray(xy1), asarray(xy2)
        radius = self.radius
        rv = xy2 - xy1
        r = norm(xy1 - xy2)
        factor = (r - 2 * radius) / r
        r0 = (xy1 + xy2) / 2.
        r1 = r0 - rv * factor / 2.
        r2 = r0 + rv * factor / 2.
        plot([r1[0], r2[0]], [r1[1], r2[1]], color='k', lw=self.lw, zorder=-1)

    def context(self, attr, val):
        '''
        change attribute in this context.

        Args:
            attr (str): the attribute to be changed.
            val (obj): the target value that used in this context.

        Return:
            obj: Context object.
        '''
        oval = getattr(self, attr)

        class Brush():
            def __enter__(brush):
                setattr(self, attr, val)

            def __exit__(brush, *args):
                setattr(self, attr, oval)
        return Brush()
