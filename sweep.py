from numpy import *
import pdb

__all__=['get_sweeper','get_psweeper','visualize_sweeper']

def get_sweeper(start,stop,nsite,iprint=1):
    '''
    Parameters:
        :start: tuple, (iteration, direction, position)
        :stop: tuple, (iteration, direction, position)
        :nsite: int, # of sites.
        :iprint: int, print level.

    Return:
        iterator,
    '''
    #validate datas
    if start[1] not in ['->','<-'] or stop[1] not in ['->','<-']:
        raise ValueError()

    #check for null iterations
    if start[2]<0 or start[2]>nsite or stop[2]<0 or stop[2]>nsite:
        return
    if stop[0]<start[0]:
        return
    elif stop[0]==start[0]:
        if stop[1]=='->' and start[1]=='<-':
            return
        elif stop[0]==start[0] and stop[1]==start[1]:
            if start[1]=='->' and stop[2]<start[2]:
                return
            elif start[1]=='<-' and stop[2]>start[2]:
                return

    direction,site=start[1],start[2]
    for iiter in xrange(start[0],stop[0]+1):
        if iprint>1:
            print '########### STARTING NEW ITERATION %s ################'%(iiter)
        while(True):
            yield iiter,direction,site
            if iiter==stop[0] and direction==stop[1] and site==stop[2]:
                return
            #next site and sweep direction
            if direction=='->':
                if site>=nsite:
                    site-=1
                    direction='<-'
                else:
                    site+=1
            else:
                if site<=0:
                    site+=1
                    direction='->'
                    break
                else:
                    site-=1
            if site<0:
                direction,site='->',0
                break
            if site>nsite:
                direction,site='<-',nsite
                break

def get_psweeper(start,stop,nsite,iprint=1):
    '''
    Get periodic sweeper.

    Parameters:
        :start: tuple, (iteration, position).
        :stop: tuple, (iteration, position).
        :nsite: int, # of sites.
        :iprint: int, print level.
    '''
    #check for null iterations
    if start[1]<0 or start[1]>nsite-1 or stop[1]<0 or stop[1]>nsite-1:
        return
    if stop[0]<start[0]:
        return
    elif stop[0]==start[0]:
        if stop[1]<start[1]:
            return

    site=start[1]
    for iiter in xrange(start[0],stop[0]+1):
        if iprint>1:
            print '########### STARTING NEW ITERATION %s ################'%(iiter)
        while(True):
            yield iiter,site
            if iiter==stop[0] and site==stop[1]:
                return
            if site>=nsite-1:
                site=0
                break
            else:
                site+=1

def visualize_sweeper(sweeper,nsite):
    import matplotlib.pyplot as plt
    dy=0.3
    arr_dx=0.3
    r=0.1
    plt.scatter(arange(nsite),zeros(nsite),s=100,facecolor='k')
    ax=plt.gca()
    ax.set_ylim(-0.5,1.)
    plt.axis('equal')
    for i,data in enumerate(sweeper):
        if i!=0:
            cc.remove()
            arr.remove()
        cc=plt.Circle((data[-1],dy),r)
        if len(data)==3:
            if data[1]=='->':
                arr=plt.Arrow(data[-1]-arr_dx-r,dy,arr_dx,0,width=0.1)
            else:
                arr=plt.Arrow(data[-1]+arr_dx+r,dy,-arr_dx,0,width=0.1)
        ax.add_patch(cc)
        ax.add_patch(arr)
        plt.draw()
        plt.pause(0.3)
