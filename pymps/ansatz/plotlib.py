import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib import patches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
from numpy.linalg import norm
import pdb

__all__ = ['draw_tensor', 'draw_kb', 'draw_rect', 'draw_tri',
           'draw_sq', 'plot_params', 'show_contract_route', 'draw_diamond']

plot_params = dict(
    legcolor='k',  # the color of legs.
    facecolor='k',  # tensor facecolor.
    legw=2,  # the width of legs.
    leglength=0.38,  # the length of legs.
    textoffset=0.12,  # the offset of texts with respect to bond.
    leg_fontsize=12,  # the fontsize of leg labels.
)


def rotate_leg(vec, theta):
    '''Rotate leg.'''
    mat = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return mat.dot(transpose(vec)).T


def _plot_textedline(pos1, pos2, label, arrow_direction=0, color=None, fontsize=None, ax=None):
    pos1, pos2 = asarray(pos1), asarray(pos2)
    leg_fontsize, legw, legcolor, textoffset, leglength = plot_params['leg_fontsize'], plot_params['legw'],\
        plot_params['legcolor'], plot_params['textoffset'], plot_params['leglength']
    if color is None:
        color = legcolor
    if fontsize is None:
        fontsize = leg_fontsize
    if ax is None:
        ax = gca()
    vec = pos2 - pos1
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], lw=legw, color=color)
    # add a text.
    # move towards the vertical direction.
    xy = (pos1 + pos2) / 2. + textoffset * array([-vec[1], vec[0]]) / norm(vec)
    ax.annotate(label, xytext=xy, xy=xy, va='center',
                ha='center', fontsize=fontsize, color=color)
    # add arrow
    if arrow_direction > 0:
        ax.arrow((pos1[0] + pos2[0]) / 2., (pos1[1] + pos2[1]) / 2., 1e-5 * (pos2[0] - pos1[0]), 1e-5 *
                 (pos2[1] - pos1[1]), color=color, head_width=0.08, head_length=0.12, length_includes_head=False)
    elif arrow_direction < 0:
        ax.arrow((pos1[0] + pos2[0]) / 2., (pos1[1] + pos2[1]) / 2., -1e-5 * (pos2[0] - pos1[0]), -1e-5 *
                 (pos2[1] - pos1[1]), color=color, head_width=0.08, head_length=0.12, length_includes_head=False)


def _plot_selfloop(pos, label, color=None, fontsize=None, ax=None, zoom=1.):
    pos = asarray(pos)
    leg_fontsize, legw, legcolor, textoffset, leglength = plot_params['leg_fontsize'], plot_params['legw'],\
        plot_params['legcolor'], plot_params['textoffset'], plot_params['leglength']
    if color is None:
        color = legcolor
    if fontsize is None:
        fontsize = leg_fontsize
    if ax is None:
        ax = gca()
    arc = patches.Arc((pos[0], pos[1] + 0.34 * zoom), width=0.2 * zoom, height=0.4 *
                      zoom, angle=0, linewidth=legw, fill=False, zorder=-2, color=color)
    ax.add_patch(arc)
    xy = pos + [0, textoffset + 0.2 * zoom + 0.34 * zoom]
    ax.annotate(label, xytext=xy, xy=xy, va='center',
                ha='center', fontsize=fontsize, color=color)


def draw_tensor(x, y, legs=[], legmask=None, flowmask=None, zoom=1., facetext=None, hatch=None, ax=None):
    '''
    Draw a general tensor, with round kernel.

    Args:
        :x,y: number, the position.
        legs (list, items are (label):theta)
        legmask (None/list): the mask array for legs.
        flowmask (None/list, the flow direction, 1 for out, -1 for in): 0 for nothing.
        zoom (number): the size of tensor.
        facetext ((str,color,number), text): color and text fontsize.
        hatch (str): matplotlib hatch.
        ax (axis): matplotlib axis.
    '''
    if ax is None:
        ax = gca()
    facecolor = plot_params['facecolor']
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leg_fontsize = plot_params['leg_fontsize']
    leglength = plot_params['leglength'] * zoom
    if legmask is None:
        legmask = [True] * len(legs)
    if flowmask is None:
        flowmask = [0] * len(legs)

    rry = rr = 0.2 * zoom  # the radius.
    rh = rv = leglength
    textoffset = plot_params['textoffset'] * zoom

    # show circle
    c1 = Circle((x, y), rr, facecolor=facecolor, hatch=hatch)
    ax.add_patch(c1)

    # draw legs, and show texts for legs
    V0 = [rr, 0]
    for leginfo, mask, flow in zip(legs, legmask, flowmask):
        label, theta = leginfo
        # draw leg
        v0 = rotate_leg(V0, theta)
        v1 = v0 * (leglength + rr) / rr
        v0, v1 = v0 + (x, y), v1 + (x, y)
        _plot_textedline(v0, v1, label, arrow_direction=flow)

    # face text
    if facetext is not None:
        t, color, size = facetext
        text(x, y, t, va='center', ha='center', color=color, fontsize=size)


def draw_kb(x, y, labels=['', '', ''], legmask=[True] * 3, flowmask=[0] * 3, hatch=None, zoom=1., ax=None, is_ket=True, facecolor=None):
    '''
    Draw a ket/bra.

    Args:
        flowmask (len-3 list, the flow direction, 1 for out, -1 for in): 0 for nothing.

    ket sticks up and bra sticks down.
    '''
    if ax is None:
        ax = gca()
    leg_fontsize = plot_params['leg_fontsize']
    facecolor = plot_params['facecolor'] if facecolor is None else facecolor
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leglength = plot_params['leglength'] * zoom

    rry = rr = 0.2 * zoom  # the radius.
    rh = rv = leglength
    textoffset = plot_params['textoffset'] * zoom
    if not is_ket:
        textoffset, rv, rry = -textoffset, -rv, -rry

    # show circle
    c1 = Circle((x, y), rr, facecolor=facecolor, hatch=hatch)
    ax.add_patch(c1)

    # draw legs, and show texts for legs
    poss = [[(x - rr, y), (x - rr - rh, y)], [(x, y + rry),
                                              (x, y + rry + rv)], [(x + rr, y), (x + rr + rh, y)]]
    for label, (pos1, pos2), mask, flow in zip(labels, poss, legmask, flowmask):
        if not mask:
            continue
        _plot_textedline(pos1, pos2, label, arrow_direction=flow)


def draw_diamond(x, y, labels=['', '', '', ''], legmask=[True] * 4, flowmask=[0] * 4, hatch=None, zoom=1., ax=None):
    '''
    Draw a diamond.

    Args:
        flowmask (len-4 list, the flow direction, 1 for out, -1 for in): 0 for nothing.
    '''
    if ax is None:
        ax = gca()
    leg_fontsize = plot_params['leg_fontsize']
    facecolor = plot_params['facecolor']
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leglength = plot_params['leglength'] * zoom
    rs = zoom * 0.15

    # draw diamond
    ax.add_patch(patches.Polygon(
        xy=[(x, y + rs), (x + rs, y), (x, y - rs), (x - rs, y)], facecolor=facecolor, hatch=hatch))
    # draw legs, and show texts for legs
    poss = [[(x - rs, y), (x - rs - leglength, y)], [(x, y + rs), (x, y + leglength + rs)],
            [(x, y - rs), (x, y - rs - leglength)], [(x + rs, y), (x + rs + leglength, y)]]
    for label, (pos1, pos2), mask, flow in zip(labels, poss, legmask, flowmask):
        if not mask:
            continue
        _plot_textedline(pos1, pos2, label, arrow_direction=flow)


def draw_rect(x, y, legs={}, hatch=None, zoom=1., ax=None, facetext=None):
    '''
    Draw a tensor.

    Args:
        :x,y: the position
        labels (list/None): the labels for each leg.
        nleg (len-2 list ):the number of legs sticking up and down.
        facetxt (tuple, (text,color):size)
    '''
    if ax is None:
        ax = gca()
    leg_fontsize = plot_params['leg_fontsize']
    facecolor = plot_params['facecolor']
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leglength = plot_params['leglength']
    width, height, pad = 1.4 * zoom, 0.6 * zoom, 0.1 * zoom
    rh = rv = leglength * zoom
    textoffset = 0.15 * zoom

    # show circle
    c1 = patches.FancyBboxPatch((x - width / 2 + pad, y - height / 2 + pad), width - 2 * pad,
                                height - 2 * pad, boxstyle='round,pad=%s' % pad, facecolor=facecolor, hatch=hatch)
    ax.add_patch(c1)

    # draw legs
    for which, leg in legs.items():
        if which == 'top':
            dy1 = height / 2
            dy2 = height / 2 + rv
            dx1, dx2 = 0, 0
        elif which == 'bottom':
            dy1 = -height / 2
            dy2 = -height / 2 - rv
            dx1, dx2 = 0, 0
        elif which == 'left':
            dy1 = dy2 = 0
            dx1 = -width / 2
            dx2 = -width / 2 - rh
        elif which == 'right':
            dy1 = dy2 = 0
            dx1 = width / 2
            dx2 = width / 2 + rh
        if hasattr(leg, '__iter__'):  # labels
            nl, labels = len(leg), leg
        else:
            nl, labels = leg, None
        x_pos = x * ones(nl)
        y_pos = y * ones(nl)
        if which == 'top' or which == 'bottom':
            x_pos += (linspace(-5. / 14 * width, 5. / 14 * width, nl)
                      if nl > 1 else array([0]))
        else:
            y_pos += (linspace(-5. / 14 * height, 5. / 14 *
                               height, nl) if nl > 1 else array([0]))
        for pi, (px, py) in enumerate(zip(x_pos, y_pos)):
            plot([px + dx1, px + dx2], [py + dy1, py + dy2],
                 color=legcolor, lw=legw)  # the vertical line
            if labels is not None:
                # show texts for legs
                xy = [px + (dx1 + dx2) / 2., py + (dy1 + dy2) / 2.]
                if which == 'top' or which == 'bottom':
                    xy[0] += textoffset
                else:
                    xy[1] -= textoffset
                annotate(str(labels[pi]), xytext=xy, xy=xy,
                         va='center', ha='center', fontsize=leg_fontsize)

    # face text
    if facetext is not None:
        t, color, size = facetext
    text(x, y, t, va='center', ha='center', color=color, fontsize=size)


def draw_tri(x, y, labels=['', '', ''], legmask=[True] * 3, flowmask=[0] * 3, hatch=None, zoom=1., ax=None, rotate=0):
    '''
    Draw a triangular tensor.

    Args:
        :x,y: float, the position.
        labels (list ,the labels for up/down,right):left
        flowmask (len-3 list, the flow direction, 1 for out, -1 for in): 0 for nothing.
    '''
    if ax is None:
        ax = gca()
    leg_fontsize = plot_params['leg_fontsize']
    facecolor = plot_params['facecolor']
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leglength = plot_params['leglength'] * zoom
    rr = 0.2 * zoom  # the radius.
    textoffset = plot_params['textoffset'] * zoom

    # show triangle
    points = rotate_leg(array([(0, rr), (rr * sqrt(3.) / 2, -rr / 2.),
                               (-rr * sqrt(3.) / 2, -rr / 2)]), rotate) + array([x, y])
    c1 = Polygon(points, facecolor=facecolor, hatch=hatch)
    ax.add_patch(c1)

    # draw legs
    pp = points[0] - [x, y]
    for i, (lbi, theta) in enumerate(zip(labels, [0, pi * 2 / 3, pi * 4 / 3])):
        if not legmask[i]:
            continue
        p1 = rotate_leg(pp, theta)
        p2 = (1 + leglength / rr) * p1
        _plot_textedline(p1 + (x, y), p2 + (x, y), lbi,
                         arrow_direction=flowmask[i])


def draw_sq(x, y, labels=[''] * 4, legmask=[True] * 4, flowmask=[0] * 4, hatch=None, zoom=1., ax=None, facecolor=None, facetext=''):
    '''
    Draw a square, cell of MPO.

    Args:
        :x,y: the position
        labels (list/None): the labels for each leg.
        legmask (len-4 list): draw legs or not.
        flowmask (len-4 list, the flow direction, 1 for out, -1 for in): 0 for nothing.
        facetext ((str,color):fontsize)
    '''
    if ax is None:
        ax = gca()
    facecolor = facecolor or plot_params['facecolor']
    legcolor = plot_params['legcolor']
    legw = plot_params['legw']
    leglength = plot_params['leglength'] * zoom
    leg_fontsize = plot_params['leg_fontsize']
    width = height = 0.4 * zoom
    textoffset = plot_params['textoffset'] * zoom

    # show square
    r1 = Rectangle((x - width / 2., y - width / 2.), width,
                   height, facecolor=facecolor, hatch=hatch)
    ax.add_patch(r1)

    # draw legs
    v1, v0 = array([-leglength - width / 2., 0]), array([-width / 2., 0])
    for i, (label, theta) in enumerate(zip(labels, [0, -pi / 2, pi / 2, pi])):
        if not legmask[i]:
            continue
        v1i, v0i = rotate_leg(v1, theta) + \
            (x, y), rotate_leg(v0, theta) + (x, y)
        _plot_textedline(v1i, v0i, label, arrow_direction=flowmask[i])

    # face text
    if facetext is not None:
        t, color, size = facetext
    text(x, y, t, va='center', ha='center', color=color, fontsize=size)

######################## Draw MPSs and TNs #################################


def show_vidalmps(vidal, offset=(0, 0)):
    '''
    Display this MPS instance graphically.

    offset:
        The global displace of the MPS chain.
    '''
    r = 0.25
    rs = 0.35
    barend = (r + rs) * 1.3
    textoffset = barend + 0.1
    color_A = 'k'
    color_S = color_A
    color_B = color_A
    edgecolor = 'k'
    facecolor = 'none'
    ax = gca()
    lc = []
    for i in range(vidal.nsite):
        xi, yi = 2 * i + offset[0], offset[1]
        ax.add_patch(patches.Circle(xy=(xi, yi), radius=r,
                                    facecolor=facecolor, edgecolor=edgecolor, hatch='xxxx'))
        lc.append([(xi, yi + r), (xi, yi + barend)])
        text(xi, yi + textoffset,
             r'$\Gamma^{\sigma_%s}$' % i, horizontalalignment='center')
        lc.append([(xi + r, yi), (xi + 1 - rs, yi)])
        ax.add_patch(patches.Polygon(xy=[(xi + 1, yi + rs), (xi + 1 + rs, yi),
                                         (xi + 1, yi - rs), (xi + 1 - rs, yi)], facecolor=facecolor, hatch='xxxx'))
        lc.append([(xi + 1, yi + rs), (xi + 1, yi + barend)])
        if i != vidal.nsite - 1:
            lc.append([(xi + 1 + rs, yi), (xi + 2 - r, yi)])
        text(xi + 1, yi + textoffset,
             r'$\Lambda^{[%s]}$' % i, horizontalalignment='center')
    lc = LineCollection(lc, color=edgecolor, lw=2)
    ax.add_collection(lc)
    axis('equal')
    ax.autoscale()


def show_mps(mps, offset=(0, 0)):
    '''
    Display this MPS instance graphically.

    Args:
        offset (tuple, (dx):dy) the global displace of the MPS chain.
    '''
    r = 0.25
    rlc = 0.5 if mps.is_ket else -0.5
    contourlw = 1
    color_A = 'none'
    color_S = color_A
    color_B = color_A
    edgecolor = '#000000'
    baroffset = r if mps.is_ket else -r
    barlen = r if mps.is_ket else -r
    textoffset = baroffset + barlen + (0.2 if mps.is_ket else -0.2)
    rowlink_offset = array([-r - 0.1, 0.1])
    collink_offset = array([0.1, r + 0.1])
    ax = gca()
    site_axis = mps.site_axis
    llink_axis = mps.llink_axis
    rlink_axis = mps.rlink_axis
    lc = []
    for i in range(mps.l):
        xi, yi = i + offset[0], 0 + offset[1]
        ax.add_patch(patches.Circle(xy=(xi, yi), radius=r, facecolor=color_A,
                                    lw=contourlw, edgecolor=edgecolor, hatch='---'))
        lc.append([(xi, yi + baroffset), (xi, yi + baroffset + barlen)])
        lc.append([(xi + r, yi), (xi + 1 - r, yi)])
        text(xi, yi + textoffset, r'$%s$' %
             mps.ML[i].labels[site_axis], horizontalalignment='center', verticalalignment='center')
        text(xi + rowlink_offset[0], yi + rowlink_offset[1], r'$%s$' %
             mps.ML[i].labels[llink_axis], horizontalalignment='center', verticalalignment='center')
        text(xi + collink_offset[0], yi + collink_offset[1], r'$%s$' %
             mps.ML[i].labels[rlink_axis], horizontalalignment='center', verticalalignment='center')
    xS, yS = mps.l + offset[0] - 0.5, offset[1]
    plot((xS, xS), (yS - 2 * r, yS + 2 * r), color=edgecolor, ls='--')
    text(xS, yS + textoffset, r'$S$',
         horizontalalignment='center', verticalalignment='center')
    for i in range(mps.nsite - mps.l):
        xi, yi = i + mps.l + offset[0], 0 + offset[1]
        ax.add_patch(patches.Circle(xy=(xi, yi), radius=r, lw=contourlw,
                                    facecolor=color_B, edgecolor=edgecolor, hatch='|||'))
        lc.append([(xi, yi + baroffset), (xi, yi + baroffset + barlen)])
        lc.append([(xi - 1 + r, yi), (xi - r, yi)])
        text(xi, yi + textoffset, r'$%s$' %
             mps.ML[mps.l + i].labels[site_axis], horizontalalignment='center', verticalalignment='center')
        text(xi + rowlink_offset[0], yi + rowlink_offset[1], r'$%s$' % mps.ML[mps.l +
                                                                              i].labels[llink_axis], horizontalalignment='center', verticalalignment='center')
        text(xi + collink_offset[0], yi + collink_offset[1], r'$%s$' % mps.ML[mps.l +
                                                                              i].labels[rlink_axis], horizontalalignment='center', verticalalignment='center')
    lc = LineCollection(lc, color=edgecolor, lw=2)
    ax.add_collection(lc)
    ax.autoscale()
    axis('equal')


def show_opc(opc, nsite=0, offset=(0, 0)):
    '''
    Show the operator graphically.

    Args:
        nsite (int): number of sites.
        offset (tuple (x,y), the offset in x):y axes.
    '''
    r = 0.25
    contourlw = 1
    facecolor = 'k'
    edgecolor = '#000000'
    textoffset = r + 0.3
    textheight = 0.7
    ax = gca()
    nnsite = 0
    for i in range(opc.nop):
        opi = opc.ops[i]
        y0 = i * textheight + offset[1] + textoffset
        nsite = max(nsite, opi.maxsite + 1)
        if isinstance(opi, OpString):
            opunits = opi.opunits
        else:
            opunits = [opi]
        for j, opij in enumerate(opunits):
            x0 = opij.siteindex + offset[0]
            text(x0, y0, opij.get_mathstr(),
                 horizontalalignment='center', verticalalignment='center')
    lc = []
    for i in range(nsite):
        xi, yi = i + offset[0], 0 + offset[1]
        lc.append([(xi - r, yi), (xi + r, yi)])
    lc = LineCollection(lc, color=edgecolor, lw=2)
    ax.add_collection(lc)
    axis('tight')
    ax.set_ylim(-1, textoffset + opc.nop * textheight)
    ax.set_xlim(-1, nsite)


def show_mpo(mpo, offset=(0, 0)):
    '''
    Display this MPO instance graphically.

    Args:
        offset (len-2 tuple): The global displace of the MPS chain.
    '''
    r = 0.25
    contourlw = 1
    facecolor = 'k'
    edgecolor = '#000000'
    baroffset = r
    barlen = r
    textoffset = baroffset + barlen + 0.2
    rowlink_offset = array([-r - 0.1, 0.1])
    collink_offset = array([0.1, r + 0.1])
    ax = gca()
    lc = []
    nsite = mpo.nsite
    for i in range(nsite):
        xi, yi = i + offset[0], 0 + offset[1]
        ax.add_patch(patches.Rectangle(xy=(xi - r, yi - r), width=2 * r,
                                       height=2 * r, facecolor=facecolor, lw=contourlw, edgecolor=edgecolor))
        lc.append([(xi, yi + baroffset), (xi, yi + baroffset + barlen)])
        text(xi, yi + textoffset, r"$%s$" %
             mpo.OL[i].labels[1], horizontalalignment='center', verticalalignment='center')
        lc.append([(xi, yi - baroffset), (xi, yi - baroffset - barlen)])
        text(xi, yi - textoffset, r"$%s$" %
             mpo.OL[i].labels[2], horizontalalignment='center', verticalalignment='center')
        text(xi + rowlink_offset[0], yi + rowlink_offset[1], r'$%s$' %
             mpo.OL[i].labels[0], horizontalalignment='center', verticalalignment='center')
        text(xi + collink_offset[0], yi + collink_offset[1], r'$%s$' %
             mpo.OL[i].labels[3], horizontalalignment='center', verticalalignment='center')
        if i != nsite - 1:
            lc.append([(xi + r, yi), (xi + 1 - r, yi)])
    lc = LineCollection(lc, color=edgecolor, lw=2)
    ax.add_collection(lc)
    ax.autoscale()
    axis('equal')


def show_opc_advanced(opc, nsite=0, offset=(0, 0)):
    '''Show the operator graphically.'''
    r = 0.2
    contourlw = 1
    facecolor = 'k'
    edgecolor = '#000000'
    textheight = 0.5
    occpunish = 0.1
    ax = gca()
    nnsite = 0
    colors = ['k', 'r', 'g', 'b']
    occ_dict = {}
    for i in range(opc.nop):
        opi = opc.ops[i]
        siteindex = opi.siteindex if hasattr(
            opi, 'opunits') else [opi.siteindex]
        y0 = offset[1]
        assert(len(siteindex) <= 2)  # only two site interaction is allowing.
        direction = 1
        if len(siteindex) == 1:
            nbody = 1
            textoffset = (r + 0.2) * direction
            xt, xd = siteindex[0] + offset[0], 0
            radi = None
        elif len(siteindex) == 2:
            nbody = 2
            x1, x2 = siteindex
            nnb = x2 - x1
            if nnb % 2 == 0:
                direction = -1
            x1, x2 = x1 + offset[0], x2 + offset[0]
            radi = (0.1 + 0.2 * nnb) * direction
            textoffset = radi + r * direction
            xt, xd = (x1 + x2) / 2., x2 - x1
        else:
            warnings.warn(
                'show_advanced is not capable for displaying more than 2-body operator.')
        nocc = occ_dict.get((xt, xd), 0)
        textoffset += (nocc * occpunish) * direction
        occ_dict[(xt, xd)] = nocc + 1
        yt = y0 + textoffset
        nsite = max(nsite, array(siteindex).max() + 1)
        text(xt, yt, opi.get_mathstr(), horizontalalignment='center',
             verticalalignment='center', fontsize=10)
        if nbody == 2:
            ax.annotate("",
                        xy=(x1, y0), xycoords='data',
                        xytext=(x2, y0), textcoords='data',
                        arrowprops=dict(arrowstyle="-",  # linestyle="dashed",
                                        color="%s" % colors[(nnb - 1) % 4],
                                        patchB=None,
                                        shrinkB=0,
                                        connectionstyle="arc3,rad=%s" % radi,
                                        ),
                        )
    lc = []
    for i in range(nsite):
        xi, yi = i + offset[0], 0 + offset[1]
        ax.add_patch(patches.Circle(xy=(xi, yi), radius=r,
                                    facecolor=facecolor, lw=contourlw, edgecolor=edgecolor))
    axis('equal')
    ax.set_xlim(-1, nsite)


def show_tnet(tnet, sites, offset=0, dangling_angles=-pi / 6):
    '''
    Show the Tensor network.

    Args:
        tnet (<TNet>):
        sites (2d array):
        offset (tuple, the x):y offset.
        dangling_angles (number): the angle for dangling legs.
    '''
    if len(sites) != tnet.ntensor:
        raise ValueError()
    ax = gca()
    leg_fontsize, legw, legcolor, textoffset, leglength = plot_params['leg_fontsize'], plot_params['legw'],\
        plot_params['legcolor'], plot_params['textoffset'], plot_params['leglength']
    sites = sites + offset
    # draw tensors
    zoom_tensor = 1.
    r = 0.2 * zoom_tensor  # 0.2 is the radius of tensor circle
    for i, (tensor, site) in enumerate(zip(tnet.tensors, sites)):
        draw_tensor(site[0], site[1], legs=[], legmask=None, hatch=None,
                    zoom=zoom_tensor, ax=ax, facetext=('$T_%s$' % i, 'w', 14))

    # draw connections.
    graph = tnet.get_connection()
    for k in range(graph.nlink):
        i, j = graph.whichlegs(k)
        label = tnet.legs[i]
        # connect sites i,j
        pos1, pos2 = sites[tnet.lid2tid(i)[0]], sites[tnet.lid2tid(j)[0]]
        vec = pos2 - pos1
        if norm(vec) == 0:
            # self loop
            _plot_selfloop(pos1, label, ax=ax)
        else:
            dv = r / norm(vec) * vec
            pos1, pos2 = pos1 + dv, pos2 - dv
            _plot_textedline(pos1, pos2, label, ax=ax)

    # draw dangling bonds.
    if ndim(dangling_angles) == 0:
        dangling_angles = dangling_angles * ones(sum(~graph.pairmask))
    for i, angle in zip(where(~graph.pairmask)[0], dangling_angles):
        leg = tnet.legs[i]
        pos0 = sites[tnet.lid2tid(i)[0]]
        dpos1 = array([0, r])
        dpos2 = dpos1 + [0, leglength * zoom_tensor]
        pos1, pos2 = pos0 + \
            rotate_leg(dpos1, angle), pos0 + rotate_leg(dpos2, angle)
        _plot_textedline(pos1, pos2, leg)


def show_contract_route(tnet, sites, order, offset=0):
    '''
    Show the route of contraction.

    Args:
        tnet (<TNet>):
        sites (2d array):
        :order: contraction order of tensors.
        offset (tuple, the x):y offset.
    '''
    color, fontsize = 'r', 20
    ax = gca()
    ntensor = tnet.ntensor
    sites = list(sites)
    for i, (it, jt) in enumerate(order):
        xy1, xy2 = sites[it] + offset, sites[jt] + offset
        _plot_textedline(xy1, xy2, label=i + 1, color='r', ax=ax)
        xy = (xy1 + xy2) / 2.
        sites[it] = xy
        sites.pop(jt)
        pause(1)
        # ax.annotate(i+1,xytext=xy,xy=xy,va='center',ha='center',fontsize=fontsize,color=color)
