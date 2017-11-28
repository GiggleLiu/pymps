from ..plt.plotlattice import LatticePloter, plt
from ..plt.plotmps import LatticePloter, plt


def plot_imp_dwave():
    ion()
    nsite = 3
    fig = figure(figsize=(5, 5))
    ploter = LatticePloter(0.2)
    ploter.set_color('k')
    for i in xrange(3):
        for j in xrange(3):
            xy = (i, j)
            ploter.plot_site(xy=xy)
            if i == 0:
                ploter.connect_sites(xy, (i - 0.7, j))
            if j == 0:
                ploter.connect_sites(xy, (i, j - 0.7))
            if i != nsite - 1:
                ploter.connect_sites(xy, (i + 1, j))
            if j != nsite - 1:
                ploter.connect_sites(xy, (i, j + 1))
            if i == nsite - 1:
                ploter.connect_sites(xy, (i + 0.7, j))
            if j == nsite - 1:
                ploter.connect_sites(xy, (i, j + 0.7))

    # show impurities
    ploter.radius = 0.13
    ploter.set_color('r')
    for xy, txt in zip([(0.5, 0.5), (1, 1), (0, 1.5)], ['A', 'B', 'C']):
        ploter.plot_site(xy=xy)
        text(xy[0] + 0.2, xy[1] + 0.2, txt,
             ha='center', va='center', fontsize=16)
    axis('equal')
    axis('off')
    tight_layout(pad=0)
    pdb.set_trace()
    savefig('imp-dwave.png')


def plot_2chain():
    ion()
    fig = figure(figsize=(7, 3), facecolor='w')
    ploter = LatticePloter(0.2)
    ploter.set_color('k')
    ploter.set_lw(4)
    nsite = 8
    y1, y2 = 0, 1.4
    for y in [y1, y2]:
        for i in xrange(nsite):
            xy = (i, y)
            ploter.plot_site(xy=xy)
            if i != nsite - 1:
                ploter.connect_sites(xy, (i + 1, y))

    # 4 boxes
    dx, dy = 0.5, 0.5
    width = 1 + 2 * dx
    height = 2 * dy
    centers = [(0.5, 0), (nsite - 1.5, 0), (0.5, y2), (nsite - 1.5, y2)]
    labels = [r'$\rho_2^L$', r'$\rho_2^R$', r'$\rho_1^L$', r'$\rho_1^R$']
    for (x0, y0), label in zip(centers, labels):
        r1 = Rectangle((x0 - width / 2., y0 - height / 2.), width,
                       height, edgecolor='none', facecolor='b', alpha=0.5, zorder=-1)
        gca().add_patch(r1)
        text(x0, y0 + height / 2. + 0.2, label,
             fontsize=16, ha='center', va='center')

    # dashed box
    for x0 in [0.5, nsite - 1.5]:
        r1 = Rectangle((x0 - width / 2. - 0.2, y2 / 2. - height / 2. - 0.9), width +
                       0.4, height + 1.8, edgecolor='b', ls='--', facecolor='none', zorder=10)
        text(x0 - 1.4, y2 / 2., r'$\rho_{1+2}^L$' if x0 < nsite /
             2 else r'$\rho_{1+2}^R$', fontsize=16, ha='center', va='center')
        gca().add_patch(r1)

    #L and R
    ccurve((nsite / 2. - 0.5, dy + y2), xscale=(nsite - 3) /
           2., yscale=0.8, ls='--', color='r')
    text(nsite / 2. - 0.5, dy + y2 + 0.5,
         r'$\rho^{L\cup R}$', fontsize=16, ha='center', va='center')

    # psi
    text(-1.4, y1, r'$\left|\psi_2\right\rangle$',
         fontsize=16, ha='center', va='center')
    text(-1.4, y2, r'$\left|\psi_1\right\rangle$',
         fontsize=16, ha='center', va='center')

    axis('equal')
    axis('off')
    xlim(-1, nsite)
    subplots_adjust(left=0, right=1)
    tight_layout()

    pdb.set_trace()
    savefig('ssf-2chain.png')


def plot_ssf_illustrate():
    ion()
    fig = figure(figsize=(7, 3), facecolor='w')
    ploter = LatticePloter(0.2)
    ploter.set_color('k')
    ploter.set_lw(4)
    nsite = 8
    y1, y2 = 0, 1.4
    for y in [y1, y2]:
        for i in xrange(nsite):
            xy = (i, y)
            ploter.plot_site(xy=xy)
            if i != nsite - 1:
                ploter.connect_sites(xy, (i + 1, y))

    # 4 boxes
    x1, x2 = 0.5, 4.5
    dx, dy = 0.3, 0.3
    width = 1 + 2 * dx
    height = 2 * dy
    centers = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    # dashed box
    for x0 in [x1, x2]:
        r1 = Rectangle((x0 - width / 2. - 0.2, y2 / 2. - height / 2. - 0.9), width + 0.4,
                       height + 1.8, edgecolor='b' if x0 < 1 else 'r', ls='--', facecolor='none', zorder=10)
        gca().add_patch(r1)

    # draw triangle
    rr = 0.2
    for x, y in [(x1, -1.), (x2, -1.)]:
        points = rotate_leg(array(
            [(0, rr), (rr * sqrt(3.) / 2, -rr / 2.), (-rr * sqrt(3.) / 2, -rr / 2)]), 0) + array([x, y])
        c1 = Polygon(points, facecolor='b' if x < 1 else 'r')
        gca().add_patch(c1)

    # psi
    text(-1.4, y1, r'$\left|\psi_2\right\rangle$',
         fontsize=16, ha='center', va='center')
    text(-1.4, y2, r'$\left|\psi_1\right\rangle$',
         fontsize=16, ha='center', va='center')

    axis('equal')
    axis('off')
    xlim(-1, nsite)
    subplots_adjust(left=0, right=1)
    tight_layout()

    pdb.set_trace()
    savefig('ssf-illustrate.png')


def plot_ssf_3type():
    ion()
    fig = figure(figsize=(7, 3.5), facecolor='w')
    ploter = LatticePloter(0.2)
    ploter.set_color('k')
    ploter.set_lw(4)
    nsite = 8
    y1, y2 = 0, 1.4
    for y in [y1, y2]:
        for i in xrange(nsite):
            xy = (i, y)
            ploter.plot_site(xy=xy)
            if i != nsite - 1:
                ploter.connect_sites(xy, (i + 1, y))

    # boxes positions
    x1, x2 = 0.5, nsite - 1.5
    dx, dy = 0.3, 0.3
    width = 1 + 2 * dx
    height = 2 * dy

    # dashed box
    for x0 in [x1, x2]:
        r1 = Rectangle((x0 - width / 2. - 0.2, y2 / 2. - height / 2. - 0.9), width + 0.4,
                       height + 1.8, edgecolor='b' if x0 < 1 else 'b', ls='--', facecolor='none', zorder=10)
        gca().add_patch(r1)

    # text F
    rr = 0.2
    xc = nsite / 2. - 0.5
    for x, y, txt in [(x1, -1., r'$F^L_{1,2}$'), (x2, -1., r'$F^R_{1,2}$'), (xc, 2.4, r'$F^{L\cup R}_{1,2}$')]:
        text(x, y, txt, va='center', ha='center', fontsize=16)

    #L and R
    ccurve((nsite / 2. - 0.5, dy + y2 + 0.2),
           xscale=(nsite - 3) / 2., yscale=0.8, ls='--', color='r')

    # psi
    text(-1.4, y1, r'$\left|\psi_2\right\rangle$',
         fontsize=16, ha='center', va='center')
    text(-1.4, y2, r'$\left|\psi_1\right\rangle$',
         fontsize=16, ha='center', va='center')

    axis('equal')
    axis('off')
    xlim(-1, nsite)
    subplots_adjust(left=0, right=1)
    tight_layout()

    pdb.set_trace()
    savefig('ssf-3type.png')


def plot_blockspin():
    ion()
    fig = figure(figsize=(4, 6), facecolor='w')
    ploter = LatticePloter(0.2)
    ploter.set_color('k')
    ploter.set_lw(4)
    nsite = 4

    text(-0.5, nsite + 0.7, '(a)', fontsize=14)
    y = nsite + 0.3
    c2 = '#888888'
    for x in xrange(nsite):
        xy = (x, y)
        ploter.set_color('k' if x % 2 == 0 else c2)
        ploter.plot_site(xy=xy)
        if x == 0:
            ploter.connect_sites(xy, (x - 0.5, y))
            ploter.connect_sites(xy, (x + 1, y))
        elif x == nsite - 1:
            ploter.connect_sites(xy, (x + 0.5, y))
        else:
            ploter.connect_sites(xy, (x + 1, y))

    text(-0.5, nsite - 0.6, '(b)', fontsize=14)
    for y in xrange(nsite):
        for x in xrange(nsite):
            xy = (x, y)
            ploter.set_color('k' if x % 2 == y % 2 else c2)
            ploter.plot_site(xy=xy)
            if x == 0:
                ploter.connect_sites(xy, (x - 0.5, y))
                ploter.connect_sites(xy, (x + 1, y))
            elif x == nsite - 1:
                ploter.connect_sites(xy, (x + 0.5, y))
            else:
                ploter.connect_sites(xy, (x + 1, y))
            if y == 0:
                ploter.connect_sites(xy, (x, y - 0.5))
                ploter.connect_sites(xy, (x, y + 1))
            elif y == nsite - 1:
                ploter.connect_sites(xy, (x, y + 0.5))
            else:
                ploter.connect_sites(xy, (x, y + 1))

    axis('equal')
    axis('off')
    xlim(-1, nsite)
    subplots_adjust(left=0, right=1)
    tight_layout()

    pdb.set_trace()
    savefig('blockspin_lattice.png')


######   Test MPS Plot  ########
mploter = MPSPloter()


def plot_mps_qflow():
    '''Quantum number flow on a MPS'''
    L = 6
    ion()
    fig = figure(figsize=(7, 1.5))
    mploter.ket(L=6, withflow=True)
    pdb.set_trace()
    savefig('mps-qflow.pdf')


def plot_mps():
    '''MPS'''
    L = 6
    ion()
    fig = figure(figsize=(7, 3))
    text(-0.7, 0.5, '(a)', fontsize=14)
    mploter.ket(L=6, withflow=False)
    dy = -1.5
    text(-0.7, 0.5 + dy, '(b)', fontsize=14)
    mploter.ket(L=6, withflow=False, offset=(0, dy))
    # link left and right
    ccurve((L / 2. - 0.5, dy), xscale=2.5, yscale=0.7,
           theta1=pi, theta2=2 * pi, color='k', lw=2)
    text(2.5, dy - 0.6, '$a_0$', ha='center', fontsize=12)

    pdb.set_trace()
    savefig('mps.pdf')


def plot_mps_qflow_():
    '''Quantum number flow on a MPS'''
    nl, nc, nr = 2, 3, 2
    dx_kb, dx_diamond = 1, 0.5
    figs = ['mps'] * nl + ['dia'] + ['mps'] * nc + ['dia'] + ['mps'] * nr
    space = array([dx_kb if fig == 'mps' else dx_diamond for fig in figs])
    space = (space[1:] + space[:-1]) / 2
    ls = range(1, nl + nc + nr + 1)
    ls.insert(nl, None)
    ls .insert(nl + nc + 1, None)
    ion()
    x, y = 0, 0
    nfig = len(figs)
    for i in xrange(nfig):
        if figs[i] == 'mps':
            flowmask = [0, -1, 1]
            if i in [3, 7, 10]:
                flowmask[0] = -1
            if i == 0:
                legmask = [0, 1, 1]
            elif i == nfig - 1:
                legmask = [1, 1, 0]
            else:
                legmask = ones(3)
            draw_kb(x, y, labels=['', '$\sigma_%s$' % (
                ls[i]), ''], legmask=legmask, flowmask=flowmask)
        else:
            draw_diamond(x, y, legmask=[False] * 4)
        if i != nfig - 1:
            x += space[i]
    axis('equal')
    pdb.set_trace()


def plot_mps_variation():
    '''Contraction for MPS'''
    ion()
    fig = figure(figsize=(7, 3))
    mploter.ket(offset=(0, 0), L=6, hatch=[
                '||'] * 2 + [''] * 2 + ['--'] * 2, withflow=True)
    mploter.mpo(offset=(0, 0.8), labelmask=[1, 0, 0, 1], L=6, withflow=True)
    mploter.bra(offset=(0, 1.6), L=6, hatch=[
                '||'] * 2 + [''] * 2 + ['--'] * 2, withflow=True)
    mploter.box(2, 3, offset=(0, 1.6), color='k')
    mploter.box(2, 3, offset=(0, 0), color='k')
    text(2.5, -0.7, r'$M$', ha='center', va='center', fontsize=16)
    text(2.5, 2.2, r'$M^*$', ha='center', va='center', fontsize=16)
    text(0.5, 2, r'$\tilde{H}$', ha='center', va='center', fontsize=16)
    xl, yl = array([0, 0, 0, 1, 1, 1, 2]), array(
        [0, 0.8, 1.6, 0, 0.8, 1.6, 0.8])
    quiver(xl[:-1], yl[:-1], diff(xl), diff(yl), color='r', lw=1,
           scale_units='xy', angles='xy', scale=1, zorder=200)
    xl, yl = array([5, 5, 5, 4, 4, 4, 3]), array(
        [0, 0.8, 1.6, 0, 0.8, 1.6, 0.8])
    quiver(xl[:-1], yl[:-1], diff(xl), diff(yl), color='r', lw=1,
           scale_units='xy', angles='xy', scale=1, zorder=200)
    text(-0.5, 0.8, r'$\tilde{H}^L$', ha='center', va='center', fontsize=16)
    text(5.5, 0.8, r'$\tilde{H}^R$', ha='center', va='center', fontsize=16)
    pdb.set_trace()
    savefig('mps-variation.pdf')


def plot_mps_state2mps():
    '''State to MPS, constructing the T matrix.'''
    ion()
    fig = figure(figsize=(7, 4))
    ax = subplot(111)
    # step1
    text(-0.7, 0.5, '(a)', fontsize=14)
    r1 = Rectangle((0, -0.2), 6, 0.4, facecolor='k', lw=2, edgecolor='k')
    ax.add_patch(r1)
    text(3, 0, '$T_1$', va='center', ha='center', fontsize=16, color='w')
    # plot ticks
    for i in xrange(6):
        xi = 0.5 + i
        plot([xi, xi], [0, 0.4], color='k', lw=2)
        text(xi, 0.6, r'$\sigma_%s$' %
             (i + 1), fontsize=12, ha='center', va='center')
    # a verticle dashed line
    plot([1, 1], [-0.5, 0.5], color='k', ls='--', lw=2)

    # step2
    # show tensor
    y0 = -1.5
    text(-0.7, y0 + 0.5, '(b)', fontsize=14)
    draw_kb(0, y0, labels=['', '$\sigma_1$', ''],
            legmask=[0, 1, 1], facecolor='w', hatch='||')
    text(0.5, y0 + 0.2, '$a_1$', fontsize=12, ha='center', va='center')
    draw_diamond(0.5, y0, legmask=[False, False, False, True])
    r1 = Rectangle((1, y0 - 0.2), 5, 0.4, facecolor='k', lw=2, edgecolor='k')
    ax.add_patch(r1)
    text(3.5, y0, "$B_1$", va='center', ha='center', fontsize=16, color='w')
    # plot ticks
    for i in xrange(1, 6):
        xi = 0.5 + i
        plot([xi, xi], [y0, y0 + 0.4], color='k', lw=2)
        text(xi, y0 + 0.6, r'$\sigma_%s$' %
             (i + 1), fontsize=12, ha='center', va='center')

    # step3
    # show tensor
    y0 = -3.
    text(-0.7, y0 + 0.5, '(c)', fontsize=14, zorder=300)
    draw_kb(0, y0, labels=['', '$\sigma_1$', ''],
            legmask=[0, 1, 1], facecolor='w', hatch='||')
    text(0.7, y0 + 0.2, '$a_1$', fontsize=12, ha='center', va='center')
    plot([0, 1], [y0, y0], color='k', lw=2, zorder=-2)
    # make it shaded
    r1 = Rectangle((-0.5, y0 - 0.75), 1., 1.5, facecolor='w',
                   lw=2, edgecolor='none', alpha=0.7, zorder=100)
    ax.add_patch(r1)

    # show T
    r1 = Rectangle((1, y0 - 0.2), 5, 0.4, facecolor='k', lw=2, edgecolor='k')
    ax.add_patch(r1)
    text(3.5, y0, "$T_2$", va='center', ha='center', fontsize=16, color='w')
    # plot ticks
    for i in xrange(1, 6):
        xi = 0.5 + i
        plot([xi, xi], [y0, y0 + 0.4], color='k', lw=2)
        text(xi, y0 + 0.6, r'$\sigma_%s$' %
             (i + 1), fontsize=12, ha='center', va='center')

    # a verticle dashed line
    plot([2, 2], [y0 - 0.5, y0 + 0.5], color='k', ls='--', lw=2)

    # step4
    # show tensor
    y0 = -4.5
    text(-0.7, y0 + 0.5, '(d)', fontsize=14)
    draw_kb(0, y0, labels=['', '$\sigma_1$', ''],
            legmask=[0, 1, 1], facecolor='w', hatch='||')
    text(0.5, y0 + 0.2, '$a_1$', fontsize=12, ha='center', va='center')
    draw_kb(1, y0, labels=['', '$\sigma_2$', ''],
            legmask=[1, 1, 1], facecolor='w', hatch='||')
    text(1.5, y0 + 0.2, '$a_2$', fontsize=12, ha='center', va='center')
    draw_diamond(1.5, y0, legmask=[False, False, False, True])
    r1 = Rectangle((2, y0 - 0.2), 4, 0.4, facecolor='k', lw=2, edgecolor='k')
    ax.add_patch(r1)
    text(3.5, y0, "$B_2$", va='center', ha='center', fontsize=16, color='w')
    # plot ticks
    for i in xrange(2, 6):
        xi = 0.5 + i
        plot([xi, xi], [y0, y0 + 0.4], color='k', lw=2)
        text(xi, y0 + 0.6, r'$\sigma_%s$' %
             (i + 1), fontsize=12, ha='center', va='center')

    axis('equal')
    axis('off')
    tight_layout(pad=0)

    pdb.set_trace()
    savefig('mps-state2mps.pdf')


def plot_mps_unitary():
    '''The unitary A and B matrices.'''
    ion()
    fig = figure(figsize=(7, 4))
    subplot(211)
    cornerlabel('(a)')
    draw_kb(0, -0.4, labels=[r'$a_{l-1}$', r'$\sigma_l$',
                             r'$a_l$'], hatch='||', facecolor='w')
    draw_kb(0, 0.4, labels=['', '', r"$a_l'$"],
            is_ket=False, hatch='||', facecolor='w')
    theta = linspace(0, pi)
    x = -0.4 * sin(theta) - 0.58
    y = 0.4 * cos(theta)
    plot(x, y, color='k', lw=2)
    text(1.4, 0, '=', ha='center', va='center', fontsize=16)
    text(0.3, 0.7, r'$A^*$', ha='center', va='center', fontsize=16)
    text(0.3, -0.7, '$A$', ha='center', va='center', fontsize=16)
    plot(x + 3.3, y, color='k', lw=2)
    text(3, 0., r'$\mathbb{1}$', ha='center', va='center', fontsize=16)
    axis('equal')
    axis('off')
    subplot(212)
    x0 = -0.6
    cornerlabel('(b)')
    draw_kb(x0, -0.4, labels=[r'$a_{l-1}$', r'$\sigma_l$',
                              r'$a_l$'], hatch='--', facecolor='w')
    draw_kb(x0, 0.4, labels=[r"$a_{l-1}'$", '', ""],
            is_ket=False, hatch='--', facecolor='w')
    theta = linspace(0, pi)
    x = 0.4 * sin(theta) + 0.58 + x0
    y = 0.4 * cos(theta)
    plot(x, y, color='k', lw=2)
    text(1.3, 0, '=', ha='center', va='center', fontsize=16)
    text(0, 0.7, r'$B^*$', ha='center', va='center', fontsize=16)
    text(0, -0.7, '$B$', ha='center', va='center', fontsize=16)
    plot(x + 2.3, y, color='k', lw=2)
    text(3, 0., r'$\mathbb{1}$', ha='center', va='center', fontsize=16)
    axis('equal')
    axis('off')

    pdb.set_trace()
    savefig('mps-unitary.pdf')


def plot_mps_scan():
    '''scan and optimize matrices.'''
    ion()
    fig = figure(figsize=(7, 2))
    mploter.ket(withflow=True)
    mploter.box(0, 1, color='r', offset=(0, -0.15))
    mploter.box(1, 2, color='b', offset=(0, -0.15))
    annotate('', (2.8, 0.8), (1.2, 0.8), arrowprops=dict(
        arrowstyle='fancy,head_width=0.6,head_length=0.8', fc='w', ec='k', connectionstyle="angle3,angleA=100,angleB=0"))
    pdb.set_trace()
    savefig('mps-scan.pdf')


def plot_mps_ssf_simple():
    '''SSF overlap matrix - the simple case.'''
    ion()
    fig = figure(figsize=(7, 2))
    mploter.ket(L=3, hatch='||', labelmask=[0, 1, 0])
    mploter.bra(L=3, offset=(0, 0.8), hatch='||', labelmask=[0, 0, 0])
    mploter.ket(L=3, offset=(3.5, 0), hatch='--',
                labelmask=[0, 1, 0], index_offset=3)
    mploter.bra(L=3, offset=(3.5, 0.8), hatch='--',
                labelmask=[0, 0, 0], index_offset=3)
    draw_diamond(2.5 + 0.25, 0, labels=['', '', '', ''], legmask=[1, 0, 0, 1])
    draw_diamond(2.5 + 0.25, 0.8,
                 labels=['', '', '', ''], legmask=[1, 0, 0, 1])
    text(2.5 + 0.3, -0.2, r'$a_3$', fontsize=12)
    text(2.5 + 0.3, 0.6, r"$a_3'$", fontsize=12)
    x = -0.5
    y = -0.4
    r1 = Rectangle((x, y), 3.7, 1.6, facecolor='none',
                   ls='--', lw=2, edgecolor='k')
    gca().add_patch(r1)
    text(-0.7, 0.8, r'$\left|\psi\right\rangle$', va='center', ha='center')
    text(-0.7, 0., r"$\left|\psi'\right\rangle$", va='center', ha='center')
    pdb.set_trace()
    savefig('mps-ssf-simple.pdf')


def plot_mps_schmidt():
    '''Schmdit decomposition.'''
    ion()
    fig = figure(figsize=(7, 2))
    mploter.ket(L=3, hatch='||', labelmask=[0, 1, 0])
    mploter.ket(L=3, offset=(3.5, 0), hatch='--',
                labelmask=[0, 1, 0], index_offset=3)
    draw_diamond(2.5 + 0.25, 0, labels=['', '', '', ''], legmask=[1, 0, 0, 1])

    # draw a half circle
    halfc((2.05, 0), xscale=0.4, yscale=0.7, color='k', lw=2, ls=':')
    halfc((3.45, 0), xscale=-0.4, yscale=0.7, color='k', lw=2, ls=':')
    text(1. + 0.3, -0.6, r'$\left|a_3\right\rangle_A$', fontsize=18)
    text(3.3 + 0.3, -0.6, r'$\left|a_3\right\rangle_B$', fontsize=18)
    pdb.set_trace()
    savefig('mps-schmidt.pdf')


def plot_mps_rho():
    '''SSF overlap matrix - the simple case.'''
    ion()
    fig = figure(figsize=(7, 2))
    mploter.ket(L=3, hatch='||', labelmask=[0, 1, 0])
    mploter.bra(L=3, offset=(0, 0.8), hatch='||', labelmask=[0, 0, 0])
    mploter.ket(L=3, offset=(3.5, 0), hatch='--',
                labelmask=[0, 1, 0], index_offset=3)
    mploter.bra(L=3, offset=(3.5, 0.8), hatch='--',
                labelmask=[0, 0, 0], index_offset=3)
    draw_diamond(2.5 + 0.25, 0, labels=['', '', '', ''], legmask=[1, 0, 0, 1])
    draw_diamond(2.5 + 0.25, 0.8,
                 labels=['', '', '', ''], legmask=[1, 0, 0, 1])
    text(2.5 + 0.3, -0.2, r'$a_3$', fontsize=12)
    text(2.5 + 0.3, 0.6, r"$a_3'$", fontsize=12)
    x = -0.5
    y = -0.4
    r1 = Rectangle((x, y), 3.7, 1.6, facecolor='none',
                   ls='--', lw=2, edgecolor='k')
    gca().add_patch(r1)
    pdb.set_trace()
    savefig('mps-rho.pdf')


def plot_mps_ssf_overlap():
    '''SSF overlap of environment blocks.'''
    ion()
    fig = figure(figsize=(7, 6))
    gs = GridSpec(5, 1)
    ax = subplot(gs[:3, :])
    cornerlabel('(a)')
    text(-0.7, 0.8, r"$\left\langle\psi\right|$",
         ha='center', va='center', fontsize=14)
    text(-0.7, 0., r"$\left|\psi\right\rangle$",
         ha='center', va='center', fontsize=14)
    mploter.ket(L=6, labelmask=[0, 1, 0])
    mploter.bra(L=6, offset=(0, 0.8), labelmask=[0, 0, 0])

    text(1.5, -0.2, r'$a_2$', fontsize=12)
    text(1.5, 0.6, r"$a_2'$", fontsize=12)
    text(4.5, -0.2, r'$a_5$', fontsize=12)
    text(4.5, 0.6, r"$a_5'$", fontsize=12)
    x = 1.5
    y = -0.4
    r1 = Rectangle((x, y), 3., 1.6, facecolor='none',
                   ls='--', lw=2, edgecolor='k')
    gca().add_patch(r1)

    # equivalent to two triangular tensor.
    text(2., -1.1, '=', fontsize=16, ha='center', va='center')
    draw_tri(3, -0.9, labels=['$k$', "$a_5'$", "$a_2'$"], zoom=1., rotate=pi)
    draw_tri(3, -1.5, labels=['', '$a_2$', '$a_5$'], zoom=1., rotate=0)
    text(4, -1.2, r'$MM^\dagger$', ha='center', va='center', fontsize=16)

    ax = subplot(gs[3:, :])
    cornerlabel('(b)')
    text(-1, 0.4, '$\mathcal{O} \;\;\;=$',
         va='center', ha='center', fontsize=16)
    mploter.ket(L=2, labelmask=[0, 1, 0])
    mploter.bra(L=2, offset=(0, 0.8), labelmask=[0, 0, 0])
    mploter.ket(L=1, offset=(2.5, 0), labelmask=[0, 1, 0], index_offset=5)
    mploter.bra(L=1, offset=(2.5, 0.8), labelmask=[0, 0, 0], index_offset=5)
    draw_tri(1.75, 1., labels=["$k'$", "$a_5'$",
                               "$a_2'$"], zoom=0.65, rotate=0)
    draw_tri(1.75, -0.2, labels=["$k$", '$a_2$',
                                 '$a_5$'], zoom=0.65, rotate=pi)
    plot([1, 1.4], [0, 0], color='k', lw=2)
    plot([1, 1.4], [0.8, 0.8], color='k', lw=2)
    plot([2.5, 2.1], [0, 0], color='k', lw=2)
    plot([2.5, 2.1], [0.8, 0.8], color='k', lw=2)
    tight_layout(pad=0)

    pdb.set_trace()
    savefig('mps-ssf-overlap.pdf')


def plot_mps_ssf_fermiop():
    '''The fermionic reordering.'''
    ion()
    fig = figure(figsize=(7, 2))
    mploter.ket()
    mploter.mpo(L=1, labelmask=[0, 1, 0, 0], offset=(0, 0.8))
    text(0.5, 1., r'$Z_1$', ha='center', va='center', fontsize=16)
    pdb.set_trace()
    savefig('mps-ssf-fermiop.pdf')


def plot_ssf_mixture():
    legsize = 16
    ion()
    fig = figure(figsize=(7, 6))
    gs = GridSpec(17, 1)
    ax = subplot(gs[:9, 0])
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
            color='k', va='top', ha='left', fontsize=14)
    # first draw sites.
    offset_L = 0
    nsite = 3
    offset_R = nsite + 1
    # the left part
    for i in xrange(nsite):
        draw_kb(i + offset_L, 0, zoom=1., is_ket=True)
    # the right part
    for i in xrange(nsite):
        draw_kb(i + offset_R, 0, zoom=1., is_ket=True)

    # draw triangular C matrices.
    ty1, ty2 = 0.29, 1.2
    draw_tri(nsite, ty1, labels=['$k$', '$a_l$', '$a_r$'], zoom=1., rotate=0)
    draw_tri(nsite, ty2, labels=['', '', ''], zoom=1., rotate=pi)
    plot([nsite - 0.2, nsite + 0.2], [(ty1 + ty2) / 2 - 0.1,
                                      (ty1 + ty2) / 2 + 0.1], color='k', ls='-')  # the cut
    # annotate it
    text(nsite + 0.4, ty1, '$M$', fontsize=14, va='center', ha='left')
    text(nsite + 0.4, ty2, '$M^{-1}$', fontsize=14, va='center', ha='left')

    # draw the overlap of system blocks.
    draw_ts(nsite, 2.2, legs={'top': ("$a_l''$", "$a_r''$"), 'bottom': [
            "$a_l'$", "$a_r'$"]}, zoom=1., facetext=('E', 'w', 16), lw=2, fontsize=legsize)

    # Annotate it
    xy1, xy2, xy3 = (0.8, 1.8), (5, 1), (-1, 0)
    annotate('Orthonormal Environment', xytext=xy1,
             xy=xy1, va='center', ha='center')
    annotate('Mixed State', xy=xy2, xytext=xy2, va='center', ha='center')
    annotate(r"$|\psi\rangle$", xy=xy3, xytext=xy3,
             va='center', ha='center', fontsize=16)

    axis([-1, 7, -0.5, 3])
    axis('equal')
    axis('off')

    # figure 2, draw overlap matrix.
    ax = subplot(gs[9:, 0], sharex=None)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            color='k', va='top', ha='left', fontsize=14)

    # the left part
    for i in xrange(nsite):
        draw_kb(i + offset_L, 0, zoom=1., is_ket=True, lw=2)
        draw_kb(i + offset_L, 1, zoom=1., is_ket=False, lw=2)
    # the right part
    for i in xrange(nsite):
        draw_kb(i + offset_R, 0, zoom=1., is_ket=True, lw=2)
        draw_kb(i + offset_R, 1, zoom=1., is_ket=False, lw=2)
    # draw C matrices.
    ty1, ty2 = -0.29, 1.29
    draw_tri(nsite, ty1, labels=['$k$', '$a_l$', '$a_r$'],
             zoom=1., reverse=True, lw=2, fontsize=legsize)
    draw_tri(nsite, ty2, labels=["$k'$", "$a_l'$", "$a_r'$"],
             zoom=1., reverse=False, lw=2, fontsize=legsize)
    text(nsite + 0.4, ty1, '$M$', fontsize=14, va='center', ha='left')
    text(nsite + 0.4, ty2, "$M'^{\dagger}$",
         fontsize=14, va='center', ha='left')
    # Annotate it
    xy1, xy2 = (-1, 1), (-1, 0)
    annotate(r"$|\psi'\rangle_S$", xytext=xy1, xy=xy1,
             va='center', ha='center', fontsize=16)
    annotate(r"$|\psi\rangle_S$", xy=xy2, xytext=xy2,
             va='center', ha='center', fontsize=16)

    axis([-1, 7, -0.5, 3])
    axis('equal')
    axis('off')
    gs.tight_layout(fig)

    pdb.set_trace()
    savefig('ssf-mixture.pdf')


def plot_nrg_addsite():
    ion()
    fig = figure(figsize=(3, 3))

    # add space, two arrows
    _plot_textedline((0, 0.2), (0, 0.5), r'$\sigma_k$',
                     arrow_direction=-1, fontsize=16)
    _plot_textedline((-0.2, 0), (-0.5, 0),
                     r'$a_{k-1}$', arrow_direction=-1, fontsize=16)

    # add a dashed curve
    ccurve((0, 0), 0.25, 0.25, theta1=0.4 * pi,
           theta2=1.1 * pi, lw=1, ls='--', color='k')
    text(-0.1, 0.1, r'$a^0_{k}$', fontsize=16, ha='center', va='center')

    axis('equal')
    xlim(-0.5, 0.2)
    axis('off')
    tight_layout()
    pdb.set_trace()
    savefig('nrg-addsite.pdf')


def plot_nrg_unitary():
    # add ket
    ion()
    fig = figure(figsize=(3, 3))

    _plot_textedline((0, 0.2), (0, 0.5), r'$\sigma_k$',
                     arrow_direction=-1, fontsize=16)
    _plot_textedline((-0.2, 0), (-0.5, 0),
                     r'$a_{k-1}$', arrow_direction=-1, fontsize=16)
    draw_kb(0, 0, zoom=1., legmask=[1, 1, 1], labels=['', '', ''], flowmask=[
            0, 0, 1], is_ket=True, facecolor='w', hatch='|')
    ccurve((0.2, 0), 0.3, 0.3, theta1=-0.3 * pi,
           theta2=0.3 * pi, lw=1, ls='--', color='k')
    text(0.35, -0.1, r'$a_{k}$', fontsize=16, ha='center', va='center')
    text(0., -0.33, r'$\bar{U}_k^\dagger$',
         fontsize=16, ha='center', va='center')

    axis('equal')
    axis('off')
    tight_layout()
    pdb.set_trace()
    savefig('nrg-unitary.pdf')


def plot_mps_nrg():
    '''Quantum number flow on a MPS'''
    L = 6
    ion()
    fig = figure(figsize=(7, 1.5))
    mploter.ket(L=6, withflow=True, hatch='||')
    _plot_textedline((5.6, 0), (5.2, 0),
                     r'$a_{6}$', arrow_direction=-1, fontsize=12)
    pdb.set_trace()
    savefig('mps-nrg.pdf')


def plot_mpo_qflow():
    '''Matrix Product Operator.'''
    L = 6
    ion()
    fig = figure(figsize=(7, 1.5))
    mploter.mpo(L=6, withflow=True)
    pdb.set_trace()
    savefig('mpo-qflow.pdf')


def plot_mps_svd():
    '''SVD decomposition.'''
    ion()
    fig = figure(figsize=(7, 4))
    # 1
    x1, y1 = 0, 0
    text(x1 - 0.7, y1 + 0.5, '1)', fontsize=14)
    draw_kb(x1, y1, labels=[r'$a_{l-1}$', r'$\sigma_l$', ''],
            flowmask=[-1, -1, 1], hatch=None, facecolor='k')
    draw_kb(x1 + 1, y1, labels=[r'$a_l$', r'$\sigma_{l+1}$', r'$a_{l+1}$'], flowmask=[
            0, -1, 1], hatch=None, facecolor='k')
    text(x1, y1 - 0.4, '$M^{\sigma_l}$', fontsize=16, ha='center', va='center')
    text(x1 + 1, y1 - 0.4, '$M^{\sigma_{l+1}}$',
         fontsize=16, ha='center', va='center')

    # 2
    x2, y2 = 3, 0
    text(x2 - 0.7, y2 + 0.5, '2)', fontsize=14)
    draw_kb(x2 + 0.5, y2, labels=[r'$a_{l-1}\sigma_l$', r'', r'$\sigma_{l+1}a_{l+1}$'], legmask=[
            1, 0, 1], flowmask=[-1, -1, 1], hatch=None, facecolor='k')
    text(x2 + 0.5, y2 - 0.4, '$T$', fontsize=16, ha='center', va='center')

    #1 & 2
    annotate('', xytext=(1.5, -0.3), xy=(2.5, -0.3), arrowprops={
             'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=-30,angleB=30', 'facecolor': 'k', 'lw': 2}, color='k')
    text(2, -0.7, u'合并指标', fontsize=14, ha='center',
         va='center', fontproperties=myfont)

    # 3
    x3, y3 = 0, -1.5
    text(x3 - 0.7, y3 + 0.5, '3)', fontsize=14)
    draw_kb(x3, y3, labels=[r'$a_{l-1}\sigma_l$', r'', r"$a_{l}'$"], legmask=[
            True, False, True], flowmask=[-1, 0, 1], hatch='||', facecolor='w')
    draw_diamond(x3 + 0.6, y3, legmask=[False] * 4)
    draw_kb(x3 + 1.2, y3, labels=[r'', r'', r'$\sigma_{l+1}a_{l+1}$'], legmask=[
            True, False, True], flowmask=[-1, 0, 1], hatch='--', facecolor='w')
    text(x3, y3 - 0.4, '$U$', fontsize=16, ha='center', va='center')
    text(x3 + 0.6, y3 - 0.4, '$S$', fontsize=16, ha='center', va='center')
    text(x3 + 1.2, y3 - 0.4, '$V$', fontsize=16, ha='center', va='center')

    # 4
    x4, y4 = 3, -1.5
    text(x4 - 0.7, y4 + 0.5, '4)', fontsize=14)
    draw_kb(x4, y4, labels=[r'$a_{l-1}$', r'$\sigma_l$', ''],
            flowmask=[-1, -1, 1], hatch='||', facecolor='w')
    draw_kb(x4 + 1, y4, labels=[r"$a_l'$", r'$\sigma_{l+1}$', r'$a_{l+1}$'], flowmask=[
            0, -1, 1], hatch=None, facecolor='k')
    text(x4, y4 - 0.4, '$A^{\sigma_l}$', fontsize=16, ha='center', va='center')
    text(x4 + 1, y4 - 0.4, '$M^{\sigma_{l+1}}$',
         fontsize=16, ha='center', va='center')

    #3 & 4
    annotate('', xytext=(1.5, -1.8), xy=(2.5, -1.8), arrowprops={
             'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=-30,angleB=30', 'facecolor': 'k', 'lw': 2}, color='k')
    text(2, -2.2, u'拆分指标', fontsize=14, ha='center',
         va='center', fontproperties=myfont)

    axis('equal')
    axis('off')
    tight_layout()
    pdb.set_trace()
    savefig('mps-svd.pdf')


def plot_mbl_ssf():
    # load data
    fname = 'ssf_6.5.dat'
    sites, fl, fr = loadtxt(fname).T

    fig = figure(figsize=(5, 3))
    plot(sites, fl, color='r', lw=2)
    plot(sites, fr, color='b', lw=2)
    # plot(sites[1:]-0.5,-diff(fl),color='b',lw=2)
    # draw ruler
    y0 = 1.
    x0 = 0.
    x1 = 0.
    anno = annotate('', xy=(0, 1.), xytext=(0, 1.), ha='center', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, lw=2, color='k'))
    plt1 = plot([x0, x0], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]
    plt2 = plot([x1, x1], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]

    xlabel('$l$')
    ylim(-0.1, 1.1)
    legend([r'$[F^L]^2$', r'$[F^R]^2$'], loc=4)
    txt = text(2, 0.1, 'Distinguishable!', alpha=0)
    tight_layout()
    nsite = len(sites)
    xlim(-1, nsite + 1)

    def update(i):
        if i >= nsite:
            sites_ = sites[::-1]
            f_ = fr[::-1]
            if i == nsite:
                txt.set_alpha(0)
            i = i - nsite
        else:
            sites_ = sites
            f_ = fl
        anno.xy = (sites_[i], f_[i])
        anno.xyann = (sites_[0], f_[i])
        if f_[i] < 1e-1:
            txt.set_alpha(1)
        plt1.set_data([sites_[0], sites_[0]], [f_[i] - 0.05, f_[i] + 0.05])
        plt2.set_data([sites_[i], sites_[i]], [f_[i] - 0.05, f_[i] + 0.05])
        return (anno, txt, plt1, plt2)
    anim = animation.FuncAnimation(gcf(), update,
                                   frames=len(sites) * 2,
                                   interval=100, repeat=False,
                                   blit=True)
    show()
    pdb.set_trace()
    anim.save('mbl-scan.gif', dpi=100, writer='imagemagick')


def plot_mps_measure():
    '''Measurement on MPS'''
    ion()
    fig = figure(figsize=(7, 3))
    mploter.ket(offset=(0, 0), L=6, hatch=[
                '||'] * 2 + [''] * 2 + ['--'] * 2, withflow=True)
    mploter.mpo(offset=(0, 0.8), labelmask=[1, 0, 0, 1], L=6, withflow=True, texts=[(r'$\mathbb{1}$', 'w', 16)] * 2 + [
                (r'$\mathcal{O}_3$', 'w', 16), (r'$\mathcal{O}_4$', 'w', 16)] + [(r'$\mathbb{1}$', 'w', 16)] * 2)
    mploter.bra(offset=(0, 1.6), L=6, hatch=[
                '||'] * 2 + [''] * 2 + ['--'] * 2, withflow=True)
    ax = gca()
    start, end, offset = 2, 3, (0, 0)
    x = offset[0] - 0.5 + start
    y = offset[1] - 0.3
    r1 = Rectangle((x, y), end - start + 1, 2.2,
                   facecolor='none', ls='--', lw=2, edgecolor='r')
    ax.add_patch(r1)
    pdb.set_trace()
    savefig('mps-measure.pdf')


def plot_spt_ssf(choice=0):
    # load data
    if choice == 0:
        fname = 'fidelity_J1.0_Jz1.0_h0_s3N120_0E0O_l.dat'
    else:
        fname = 'fidelity_J1.0_Jz1.0_h0_Uzz-0.5_s3N120_0E0O_l.dat'
    fr = loadtxt(fname)
    sites = arange(len(fr))
    fl = fr[::-1]

    fig = figure(figsize=(5, 3))
    plot(sites, fl, color='r', lw=2)
    plot(sites, fr, color='b', lw=2)
    # plot(sites[1:]-0.5,-diff(fl),color='b',lw=2)
    # draw ruler
    y0 = 1.
    x0 = 0.
    x1 = 0.
    anno = annotate('', xy=(0, 1.), xytext=(0, 1.), ha='center', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, lw=2, color='k'))
    plt1 = plot([x0, x0], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]
    plt2 = plot([x1, x1], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]

    xlabel('$l$')
    ylim(-0.1, 1.1)
    legend([r'$[F^L]^2$', r'$[F^R]^2$'], loc=4, frameon=False)
    txt = text(2, 0.1, 'Distinguishable!', alpha=0)
    tight_layout()
    nsite = len(sites)
    xlim(-1, nsite + 1)

    def update(i):
        i = i * 2
        if (i >= nsite and i < nsite + 8) or i >= 2 * nsite + 8:
            return ()
        if i >= nsite:
            i = i - 8
            sites_ = sites[::-1]
            f_ = fr[::-1]
            if i == nsite + 1:
                txt.set_alpha(0)
            i = i - nsite + 1
        else:
            sites_ = sites
            f_ = fl
        anno.xy = (sites_[i], f_[i])
        anno.xyann = (sites_[0], f_[i])
        if f_[i] < 1e-1:
            txt.set_alpha(1)
        plt1.set_data([sites_[0], sites_[0]], [f_[i] - 0.05, f_[i] + 0.05])
        plt2.set_data([sites_[i], sites_[i]], [f_[i] - 0.05, f_[i] + 0.05])
        # anim.event_source.interval=i**2/100.
        if i == nsite - 1:
            anim.event_source.interval = 1000
        else:
            anim.event_source.interval = i
        return (anno, txt, plt1, plt2)
    anim = animation.FuncAnimation(gcf(), update,
                                   frames=nsite + 8,
                                   interval=50, repeat=False,
                                   blit=False)
    # show()
    anim.save('spt-scan%s.gif' % choice, dpi=100, writer='imagemagick')


def plot_spt_ssf_edge(choice=0):
    # load data
    if choice == 0:
        fname1 = 'fidelity_J1.0_Jz1.0_h0_s3N120_0E0O_eFalse.dat'
    elif choice == 1:
        fname1 = 'fidelity_J1.0_Jz1.0_h0_Uzz-0.5_s3N120_0E0O_eFalse.dat'
    elif choice == 2:
        fname1 = 'fidelity_D0.5_mu2.3_V0.6_N120_01_e.dat'
    elif choice == 3:
        fname1 = 'fidelity_D0.5_mu2.3_V0.6_N120_01_eFalse.dat'
    f1 = loadtxt(fname1)[::-1]**2
    sites = 2 * arange(len(f1))

    fig = figure(figsize=(5, 3))
    plot(sites, f1, color='r', lw=2)
    # draw ruler
    y0 = 1.
    x0 = 0.
    x1 = 0.
    anno = annotate('', xy=(0, 1.), xytext=(0, 1.), ha='center', va='center', fontsize=14,
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, lw=2, color='k'))
    plt1 = plot([x0, x0], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]
    plt2 = plot([x1, x1], [y0 - 0.05, y0 + 0.05], color='k', lw=2)[0]

    xlabel('$l$')
    ylim(-0.1, 1.1)
    legend([r'$[F^{\bar{C}}_{\rm SPT}]^2$', r'$[F^{\bar{C}}_{\rm CAT}]^2$', r'$[F^{\bar{C}}_{\rm Kitaev}]^2$',
            r'$[F^{\bar{C}}_{\rm XZ}]^2$'][choice:choice + 1], loc=4, frameon=False)
    txt = text(2, 0.1, 'Distinguishable!', alpha=0)
    tight_layout()
    nsite = len(sites)
    xlim(-1, nsite * 2 + 1)

    def update(i):
        if i >= nsite:
            return ()
        f_ = f1
        anno.xy = (sites[i], f_[i])
        anno.xyann = (sites[0], f_[i])
        if f_[i] < 1e-1:
            txt.set_alpha(1)
        plt1.set_data([sites[0], sites[0]], [f_[i] - 0.05, f_[i] + 0.05])
        plt2.set_data([sites[i], sites[i]], [f_[i] - 0.05, f_[i] + 0.05])
        # anim.event_source.interval=i**2/100.
        if i == nsite - 1:
            anim.event_source.interval = 1000
        else:
            anim.event_source.interval = i
        return (anno, txt, plt1, plt2)
    anim = animation.FuncAnimation(gcf(), update,
                                   frames=nsite + 8,
                                   interval=50, repeat=False,
                                   blit=False)
    # show()
    anim.save('spt-scan-edge%s.gif' % choice, dpi=100, writer='imagemagick')


# plot_mps_qflow()
# plot_mps()
# plot_mpo_qflow()
# plot_mps_scan()
# plot_mps_variation()
# plot_mps_measure()
# plot_mps_ssf_fermiop()
# plot_mps_unitary()
# plot_mps_rho()
# plot_mps_ssf_overlap()
# plot_mps_ssf_simple()
# plot_mps_state2mps()
# plot_mps_schmidt()
# plot_nrg_addsite()
# plot_nrg_unitary()
# plot_mps_nrg()
# plot_mps_svd()
# plot_ssf_mixture()
# plot_spt_ssf()
plot_spt_ssf_edge(2)
# plot_mbl_ssf()
# plot_blockspin()

# plot_imp_dwave()
plot_2chain()
# plot_ssf_illustrate()
# plot_ssf_3type()
