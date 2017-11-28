try:
    from matplotlib import pyplot as plt
    import matplotlib
except:
    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    from matplotlib import pyplot as plt
import numpy as np
import pdb

__all__ = ['DataPlt', 'NoBoxPlt', 'matplotlib', 'plt']


class DataPlt():
    '''
    Dynamic plot context, intended for displaying geometries.
    like removing axes, equal axis, dynamically tune your figure and save it.

    Args:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.

    Attributes:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.
        ax (Axes): matplotlib Axes instance.

    Examples:
        with DynamicShow() as ds:
            c = Circle([2, 2], radius=1.0)
            ds.ax.add_patch(c)
    '''

    def __init__(self, figsize=(6, 4), filename=None, dpi=300):
        self.figsize = figsize
        self.filename = filename
        self.ax = None

    def __enter__(self):
        _setup_mpl()
        plt.ion()
        plt.figure(figsize=self.figsize)
        self.ax = plt.gca()
        return self

    def __exit__(self, *args):
        plt.tight_layout()
        if self.filename is not None:
            print('Press `c` to save figure to "%s", `Ctrl+d` to break >>' %
                  self.filename)
            pdb.set_trace()
            plt.savefig(self.filename, dpi=300)
        else:
            pdb.set_trace()


class NoBoxPlt():
    '''
    Dynamic plot context, intended for displaying geometries.
    like removing axes, equal axis, dynamically tune your figure and save it.

    Args:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.

    Attributes:
        figsize (tuple, default=(6,4)): figure size.
        graph_layout (tuple|None): number of graphs, None for single graph.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.
        ax (Axes): matplotlib Axes instance.

    Examples:
        with DynamicShow() as ds:
            c = Circle([2, 2], radius=1.0)
            ds.ax.add_patch(c)
    '''

    def __init__(self, figsize=(6, 4), graph_layout=None, filename=None, dpi=300):
        self.figsize = figsize
        self.filename = filename
        self.ax = None
        self.graph_layout = graph_layout

    def __enter__(self):
        _setup_mpl()
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        if self.graph_layout is None:
            self.ax = plt.subplot(111)
        else:
            self.ax = []
            self.gs = plt.GridSpec(*self.graph_layout)
            for i in range(self.graph_layout[0]):
                for j in range(self.graph_layout[1]):
                    self.ax.append(plt.subplot(self.gs[i, j]))
        return self

    def __exit__(self, *args):
        axes = [self.ax] if self.graph_layout is None else self.ax
        for ax in axes:
            ax.axis('equal')
            ax.axis('off')
        plt.tight_layout()
        if self.filename is not None:
            print('Press `c` to save figure to "%s", `Ctrl+d` to break >>' %
                  self.filename)
            pdb.set_trace()
            plt.savefig(self.filename, dpi=300)
        else:
            pdb.set_trace()


def _setup_mpl():
    '''customize matplotlib.'''
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 18


def _setup_font():
    myfont = matplotlib.font_manager.FontProperties(
        family='wqy', fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    matplotlib.rcParams["pdf.fonttype"] = 42
    return myfont
