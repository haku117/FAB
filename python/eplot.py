
import matplotlib
matplotlib.use('Agg')

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys


ln_style =['', '-', '--', '-.', ':']

fdir = '/home/haku7117/'

def drawListCmp(prefix, mList1, mList2, midfix, suffix, w, n=200):
    # assert(len(mList1) == len(mList2))
    plt.figure()
    l1 = len(mList1)
    l2 = len(mList2)
    lgd=[]
    suffix += '.txt'

    for i in range(l1):
        for j in range(l2):
            f_name = prefix + mList1[i] + midifx + mList2[j] + suffix
            print(os.stat(f_name).st_size)
            if os.stat(f_name).st_size > 0:
                d1=pandas.read_csv(f_name,skiprows=0, header=None)
                #print(d1)
                plt.plot(d1[:n][0], d1[:n][1], linestyle = ln_style[j])
                lgd.append(mList1[i] + 'k-' + mList2[j])
    #plt.hold(True)
    plt.legend(lgd)
    plt.xlabel('time (s)')
    plt.ylabel('objective function value')
    plt.title('NMF Convergence speed comparison with batch size')
    #plt.savefig(prefix.replace('-100k/','-').replac..e('/','')+'.png')
    plt.savefig(fdir + 'FAB/fig/nmf-bs30k-w64.pdf')
    plt.show()

drawListCmp(fdir + 'results/score/1000-3k/', '', ['30','60'],['dcfsb','dcgrp'], 
    '0000-0.00001/', '-64-125', 64)

