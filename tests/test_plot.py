from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
sys.path.insert(0,'../')

from plotlib import *


def test_drawketbra():
    ion()
    for i in xrange(5):
        draw_kb(i,2,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i],zoom=1.,is_ket=True)
        draw_kb(i,5,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i],zoom=1.2,is_ket=False)
    axis('equal')

def test_drawtensor():
    ion()
    leginfo=[('a',pi),('b',-0.5*pi),('c',0)]
    for i in xrange(5):
        draw_tensor(i*2,2,legs=[('$%s_%s$'%(tk,i),pi/10*i+dth) for tk,dth in leginfo],legmask=[True,True,True],hatch=None,zoom=1.,ax=None)
        draw_tensor(i*2,5,legs=[('$%s_%s$'%(tk,i),-pi/10*i+dth) for tk,dth in leginfo],legmask=[True,False,True],hatch='///',zoom=1.2,ax=None)
    axis('equal')

def test_drawts():
    ion()
    for i in xrange(5):
        draw_rect(i*3,2,legs={'top':('$a_%s$'%i,'$b_%s$'%i),'left':(0,1,2),'right':random.randint(3),'bottom':['$c_%s$'%i]*random.randint(1,5)},zoom=1.,facetext=('A','w',16))
    axis('equal')

def test_drawtri():
    ion()
    for i in xrange(5):
        draw_tri(i,2,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i],zoom=1.,rotate=0)
        draw_tri(i,5,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i],zoom=1.2,rotate=i/6.*pi)
    axis('equal')

def test_drawsq():
    ion()
    for i in xrange(5):
        draw_sq(i,2,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i,''],legmask=[True,True,False,False],zoom=1.)
        draw_sq(i,5,labels=['$a_%s$'%i,'$b_%s$'%i,'$c_%s$'%i,''],legmask=[False,True,True,True],zoom=1.2)
    axis('equal')

def test_mixture():
    legsize=16
    ion()
    fig=figure(figsize=(7,6),facecolor='w')
    gs=GridSpec(17,1)
    ax=subplot(gs[:9,0])
    ax.text(0.02,0.95,'(a)',transform=ax.transAxes,color='k',va='top',ha='left',fontsize=14)
    #first draw sites.
    offset_L=0
    nsite=3
    offset_R=nsite+1
    #the left part
    for i in xrange(nsite):
        draw_kb(i+offset_L,0,zoom=1.,is_ket=True)
    #the right part
    for i in xrange(nsite):
        draw_kb(i+offset_R,0,zoom=1.,is_ket=True)

    #draw triangular C matrices.
    ty1,ty2=0.29,1.2
    draw_tri(nsite,ty1,labels=['$k$','$a_l$','$a_r$'],zoom=1.,rotate=0,fontsize=legsize)
    draw_tri(nsite,ty2,labels=['','',''],zoom=1.,rotate=pi)
    plot([nsite-0.2,nsite+0.2],[(ty1+ty2)/2-0.1,(ty1+ty2)/2+0.1],color='k',ls='-')  #the cut
    #annotate it
    text(nsite+0.4,ty1,'$C$',fontsize=14,va='center',ha='left')
    text(nsite+0.4,ty2,'$C^{-1}$',fontsize=14,va='center',ha='left')

    #draw the overlap of system blocks.
    draw_rect(nsite,2.2,legs={'top':("$a_l''$","$a_r''$"),'bottom':["$a_l'$","$a_r'$"]},zoom=1.,facetext=('E','w',16),fontsize=legsize)

    #Annotate it
    xy1,xy2,xy3=(0.8,1.8),(5,1),(-1,0)
    annotate('Orthogonal Environment',xytext=xy1,xy=xy1,va='center',ha='center')
    annotate('Mixture',xy=xy2,xytext=xy2,va='center',ha='center')
    annotate(r"$|\psi\rangle$",xy=xy3,xytext=xy3,va='center',ha='center',fontsize=16)

    axis([-1,7,-0.5,3])
    axis('equal')
    axis('off')

    #figure 2, draw overlap matrix.
    ax=subplot(gs[9:,0],sharex=None)
    ax.text(0.02,0.95,'(b)',transform=ax.transAxes,color='k',va='top',ha='left',fontsize=14)

    #the left part
    for i in xrange(nsite):
        draw_kb(i+offset_L,0,zoom=1.,is_ket=True)
        draw_kb(i+offset_L,1,zoom=1.,is_ket=False)
    #the right part
    for i in xrange(nsite):
        draw_kb(i+offset_R,0,zoom=1.,is_ket=True)
        draw_kb(i+offset_R,1,zoom=1.,is_ket=False)
    #draw C matrices.
    ty1,ty2=-0.29,1.29
    draw_tri(nsite,ty1,labels=['$k$','$a_l$','$a_r$'],zoom=1.,rotate=pi,fontsize=legsize)
    draw_tri(nsite,ty2,labels=["$k'$","$a_l'$","$a_r'$"],zoom=1.,rotate=0,fontsize=legsize)
    text(nsite+0.4,ty1,'$C$',fontsize=14,va='center',ha='left')
    text(nsite+0.4,ty2,"$C'^{\dagger}$",fontsize=14,va='center',ha='left')
    #Annotate it
    xy1,xy2=(-1,1),(-1,0)
    annotate(r"$|\psi'\rangle_S$",xytext=xy1,xy=xy1,va='center',ha='center',fontsize=16)
    annotate(r"$|\psi\rangle_S$",xy=xy2,xytext=xy2,va='center',ha='center',fontsize=16)

    axis([-1,7,-0.5,3])
    axis('equal')
    axis('off')
    gs.tight_layout(fig)

def test_all():
    test_drawketbra()
    pause(1)
    cla()
    test_drawts()
    pause(1)
    cla()
    test_drawtri()
    pause(1)
    cla()
    test_drawsq()
    pause(1)
    cla()
    test_drawtensor()
    pause(1)
    #plot_mixture()

if __name__=='__main__':
    test_all()
