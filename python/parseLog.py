import os.path
import numpy as np

def loadFile(file_name, nWorker):

    file = open(file_name, 'r')
    line_array = file.readlines()

    totalCalT = 0
    totalWaitT = 0
    stat = 0
    for line in line_array:
        if '(W' in line and 'Statistics' in line:
            stat = 1
        elif stat > 0:
            stat += 1
            if stat == 4:
                lst = line.split('\t')
                # print(lst)
                iterNum = lst[0].split(' ')[1]
                totalCalT += float(lst[2].split(' ')[1])
                # print(iterNum, calT)
            elif stat == 5:
                lst = line.split('\t')
                # print(lst)
                totalWaitT += float(lst[3].split(' ')[1])
                stat = 0

    return (iterNum, "{:.3f}".format(totalCalT/nWorker), "{:.3f}".format(totalWaitT/nWorker))

mode = ['fsb', 'dcfsb']
wrk = [2, 4, 8, 16]

matrix = np.zeros((5, 5))

dir = '../build/log/10-1m/'
path = '000-0.01/'

for m in mode:
    for w in wrk:
        file_name = m + '-' + str(w)
        if os.path.isfile(file_name):
            print(file_name, loadFile(file_name, w))
