# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib
matplotlib.use('Agg')

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys

os.chdir('/home/gzhao/mnil/FAB/build/figure/')
#os.chdir('E:/Code/FSB/score/10-100k/1000-0.1')
#os.chdir('E:/Code/FSB/score/10,15,1-100k/1000-0.1')

USE_HEADER=True;

def drawOne(mode, nw, n=None):
    d = pandas.read_csv(mode+'-'+str(nw)+'.txt',header=None);
    plt.plot(d[:n][0], d[:n][1])
    plt.legend(mode+'-'+str(nw))
    plt.show()

#drawOne('sync', 1)
#drawOne('fab', 1)

def drawGroup(mode, rng, n):
    #plt.hold(True)
    for i in rng:
        d = pandas.read_csv(mode+'-'+str(i)+'.txt',skiprows=0, header=None);
        plt.plot(d[:][0], d[:][1])
    plt.legend([str(i) for i in rng])
    plt.title(mode)
    plt.show()
    #plt.hold(False)

#rng=list(range(1,7))
rng=[1, 2, 4, 8]

#drawGroup('sync', rng)
#drawGroup('fsb', rng)
#drawGroup('async', rng)
#drawGroup('fab', rng)


def drawCmp(mode1, mode2, nw, n=200):
    plt.figure();
    d1=pandas.read_csv(mode1+'-'+str(nw)+'.txt',skiprows=0, header=None);
    d2=pandas.read_csv(mode2+'-'+str(nw)+'.txt',skiprows=0, header=None);
    #plt.hold(True)
    plt.plot(d1[:n][0], d1[:n][1])
    plt.plot(d2[:n][0], d2[:n][1])
    plt.legend([mode1, mode2])
    plt.show()
    
    
#drawCmp('async','fsb',8)
#drawCmp('sync','fsb',1)
#drawCmp('sync','fsb',2)
#drawCmp('sync','fsb',4)
#drawCmp('sync','fsb',8,300)
#drawCmp('async','fab',1)
#drawCmp('async','fab',2)
#drawCmp('async','fab',2,20)

def drawList(prefix, mList, w, bs, n=200):
    plt.figure();
    for m in mList:
        d=pandas.read_csv(prefix+m+'.txt',skiprows=0, header=None)
        plt.plot(d[:n][0], d[:n][1])
    #plt.hold(True)
    plt.legend(mList)
    plt.show()
    plt.savefig('/home/gzhao/mnil/boxCOM/figure/fsb-bs'+ bs +'k-r01-w'+ w +'.pdf')

print('input para:',sys.argv)
if len(sys.argv)>2:
    w=sys.argv[1]
    bs=sys.argv[2]
#drawList('../score/10-1m/'+ bs + '000-0.01/',['fsb-'+w,'dcfsb-'+w], w, bs)
#drawList('../score/10-1m/'+ bs + '000-0.01/',['fsb-2','fsb-4','fsb-8','dcfsb-2','dcfsb-4','dcfsb-8'], w, bs)
#drawList('../10000-0.01/',['fab-1','fab-2','fab-4','fab-8'],10)
#drawList('../score/10-1m/'+ '20000-0.01/',['dcfsb-2','dcfsb-4','dcfsb-8','dcfsb-16'],'20','x')
#drawList('../score/10-1m/'+ bs + '000-0.01/',['fsb-'+w,'dcfsb-'+w, 'async-'+w], w, bs)
#drawList('10000-0.1/',['sync-4','fsb-4','async-4','fab-4'])

def drawListCmp(prefix, mList1, mList2, bs, n=200):
    assert(len(mList1) == len(mList2))
    plt.figure()
    l=len(mList1)
    lgd=[]
    for i in range(l):
        print(os.stat(prefix+mList1[i]+'.txt').st_size)
        if os.stat(prefix+mList1[i]+'.txt').st_size > 0:
            d1=pandas.read_csv(prefix+mList1[i]+'.txt',skiprows=0, header=None)
            #print(d1)
            plt.plot(d1[:n][0], d1[:n][1])
            lgd.append(mList1[i])
        print( os.stat(prefix+mList2[i]+'.txt').st_size)
        if os.stat(prefix+mList2[i]+'.txt').st_size > 0:
            d2=pandas.read_csv(prefix+mList2[i]+'.txt',skiprows=0, header=None)
            #print(d2)
            plt.plot(d2[:n][0], d2[:n][1], '--')
            lgd.append(mList2[i])
    #plt.hold(True)
    plt.legend(lgd)
    plt.xlabel('time (s)')
    plt.ylabel('objective function value')
    plt.title('MLP Convergence speed comparison with batch size '+bs+'k' )
    #plt.savefig(prefix.replace('-100k/','-').replace('/','')+'.png')
    plt.savefig('/home/gzhao/mnil/boxCOM/figure/mlp-bs'+ bs +'k-r01-wxx.pdf')
    plt.show()

#drawListCmp('../score/10-1m/'+ bs + '000-0.01/',['fsb-2','fsb-4','fsb-8'],['dcfsb-2','dcfsb-4','dcfsb-8'], bs)
drawListCmp('../score/10,15,1-100k/'+ bs + '000-0.01/',['fsb-2','fsb-4','fsb-8','fsb-16'],['dcfsb-2','dcfsb-4','dcfsb-8','dcfsb-16'], bs)
#drawListCmp('../10000-0.1/',['async-1','async-2','async-4'],['fab-1','fab-2','fab-4'])
#drawListCmp('10,15,1-100k/1000-0.1/',['async-1','async-2','async-4', 'async-8'],['fab-1','fab-2','fab-4','fab-8'])

# returnn time and score, each file holds a ROW
def getRecord(prefix, mList, asMatrix=False):
    time=[]
    score=[]
    for m in mList:
        t=pandas.read_csv(prefix+m+'.txt',skiprows=0, header=None)
        time.append(t[:][0].values)
        score.append(t[:][1].values)
    if asMatrix == True:
        time=np.array(time)
        score=np.array(score)
    return time,score


def score2progress(score, p0=None, pinf=None):
    if isinstance(score, np.ndarray):
        # 1-D and 2-D np.array
        if p0 is None:
            p0=score.max()
        if pinf is None:
            pinf=score.min()
        return (p0-score)/(p0-pinf)
    else:
        # list of np.array
        n=len(score)
        if p0 is None or pinf is None:
            tp0=0
            tpinf=np.inf
            for i in range(n):
                tp0=max(tp0, score[i].max())
                tpinf=min(tpinf, score[i].min())
            if p0 is None:
                p0=tp0
            if pinf is None:
                pinf=tpinf
        res=[]
        for i in range(n):
            res.append((p0-score[i])/(p0-pinf))
        return res

def findTime4score(time, score, spoint):
    if isinstance(score, np.ndarray) and score.ndim == 2:
        # matrix
        t=np.abs(score-spoint)
        idx=t.argmin(1)
    elif isinstance(score, list) and isinstance(score[0], np.ndarray):
        # list of np.array
        idx=[]
        for sl in score:
            t=np.abs(sl-spoint)
            idx.append(t.argmin())
    else: # isinstance(score, numpy.ndarray) && score.ndim == 1:
        # 1-D array
        t=np.abs(score-spoint)
        idx=t.argmin()
        return time[idx]
    res=[]
    for i in range(len(idx)):
        res.append(time[i][idx[i]])
    return np.array(res)


def drawScale(time, score, x, ppoints, refX=None):
    assert len(time) == len(x) and len(score) == len(x)
    if refX==None:
        refIdx=0
    else:
        refIdx=x.index(refX)
    factor = np.array(x).reshape([len(x),1]) / x[refIdx]
    progress=score2progress(score)
    lgd=[]
    plt.figure()
    for pp in ppoints:
        t = findTime4score(time, progress, pp)
        plt.plot(x, t, '-')
        plt.plot(x, t[refIdx]/factor, '--')
        lgd.append(str(pp)+'-act')
        lgd.append(str(pp)+'-opt')
    plt.xlabel('# of workers')
    plt.ylabel('time (s)')
    plt.legend(lgd)
    plt.show()
    
    
#time,score=getRecord('../10000-0.1/',['fab-1','fab-2','fab-4','fab-8'])
#drawScale(time, score, [1,2,4,8], [0.9])


def drawScaleCmpAll(prefix, modes, x, ppoint, showRef=False):
    def getNameList(head, rng):
        return [head+str(i) for i in rng]
    time=[]; score=[];
    for m in modes:
        t,s=getRecord(prefix, getNameList(m+'-', x))
        time.append(t)
        score.append(s)
    
    def getMinMax(scoreList):
        mi = min([sl.min() for sl in scoreList])
        ma = max([sl.max() for sl in scoreList])
        return mi,ma
    s0=0; sinf=np.inf
    for s in score:
        mi,ma=getMinMax(s)
        s0=max(s0,ma)
        sinf=min(sinf,mi)
    
    progress=[]
    for s in score:
        p=score2progress(s, s0, sinf)
        progress.append(p)
    
    plt.figure()
    factor = np.array(x) / x[0]
    lgd=[]
    for i in range(len(modes)):
        t = findTime4score(time[i], progress[i], ppoint)
        plt.plot(x, t)
        lgd.append(modes[i])
        if showRef:
            plt.plot(x, t[0]/factor, 'k--')
            lgd.append(modes[i]+'-ref')
    plt.legend(lgd)
    plt.show()

#drawScaleCmpAll('../1000-0.1/', ['sync','fsb','async','fab'], [1,2,4,8], 0.85)

    
