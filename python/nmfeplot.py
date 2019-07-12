
import matplotlib
matplotlib.use('Agg')

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FuncFormatter


ln_style =['-', '--', '-.', ':', '-', '--', '-.', ':']

def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    #print val, signf
                    if str(val).split(".")[1] == "0":
                        return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])
    return y

fdir = '../build/'

def drawListCmp(prefix, mList1, mList2, midfix, suffix, w, n=200):
    # assert(len(mList1) == len(mList2))
    plt.figure()
    l1 = len(mList1)
    l2 = len(mList2)
    lgd=[]
    suffix += '.txt'

    for i in range(l1):
        for j in range(l2):
            f_name = prefix + mList1[i] + midfix + mList2[j] + suffix
            print(mList1[i] + '/' + mList2[j])
            if not os.path.isfile(f_name):
                continue
            print(os.stat(f_name).st_size)
            if os.stat(f_name).st_size > 0:
                d1=pandas.read_csv(f_name,skiprows=0, header=None)
                #print(d1)
                plt.plot(d1[:n][0], -d1[:n][1], linestyle = ln_style[j])
                lgd.append(mList1[i][:len(mList1[i])-2] + '-' + mList2[j])
    #plt.hold(True)
    plt.legend(lgd)
    plt.xlabel('time (s)')
    plt.ylabel('objective function value')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_fmt))
    plt.title('LDA Convergence speed comparison with batch size')
    #plt.savefig(prefix.replace('-100k/','-').replac..e('/','')+'.png')
    plt.savefig(fdir + '../../fig/lda-bsx-k100-w' + str(w) + '.pdf')
    plt.show()

drawListCmp(fdir + 'score/1000-2k/', ['200000', '100000','50000'],['sync','async'], 
    '-0.00001/', '-4-100', 4)

