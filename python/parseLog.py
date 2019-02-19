import os.path
import numpy as np
import math


def loadFile(file_name, nWorker):

    file = open(file_name, 'r')
    line_array = file.readlines()

    totalCalT = 0
    totalWaitT = 0
    tt_dp = 0
    iterNum = 0
    stat = 0
    for line in line_array:
        if '(W' in line and 'Statistics' in line:
            stat = 1
        elif stat > 0:
            stat += 1
            # if stat == 4:
            #     lst = line.split('\t')
            #     # print(lst)
            #     iterNum = lst[0].split(' ')[1]
            #     totalCalT += float(lst[2].split(' ')[1])
            #     # print(iterNum, calT)
            # elif stat == 5:
            #     lst = line.split('\t')
            #     # print(lst)
            #     totalWaitT += float(lst[3].split(' ')[1])
            #     stat = 0
            if stat == 6:
                lst = line.split('\t')
                iterNum = lst[0].split(' ')[1]
                tt_dp += lst[1].split(' ')[1]
                totalCalT += float(lst[2].split(' ')[1])
                totalWaitT += float(lst[3].split(' ')[1])
                stat = 0

    return(iterNum, "{:.3f}".format(totalCalT/nWorker), "{:.3f}".format(totalWaitT/nWorker), tt_dp)


output_name = '../../boxCOM/figure/resultMatrixKM.txt'
output = open(output_name, 'w')
#output.write("what")

mode = ['dcfsb']
wrk = [2, 4, 8, 16]
bs = ['2']#, '5', '10', '20']
catagrory = ['iter', '#dp' 'calT', 'waitT']

dim = len(mode) * len(bs) * len(catagrory)
matrix = np.zeros((dim, 4))

titleW = ''
for i in range(1, 5):
    titleW += '\tw'+str(int(2 ** i))

dir = '../build/log/10-100k/'
path = '000-1/'

for s in range(len(bs)):
    for m in range(len(mode)):
        for w in range(len(wrk)):
            wNum = int(2**(w+1))
            file_name = dir + bs[s] + path + mode[m] + '-' + str(wNum)
            if os.path.isfile(file_name):
                result = loadFile(file_name, wNum)
                print(file_name, result)
                for j in range(len(result)):
                    matrix[s + len(result)*j + len(catagrory)*len(bs)*m][w] = result[j]

print(matrix)
#output.write(matrix)

for m in range(len(mode)):
    for i in range(len(catagrory)):
        title = mode[m] + '-' + catagrory[i] + titleW
        output.write(title + '\n')
        for j in range(len(bs)):
            line = bs[j]+'k'
            for w in range(len(wrk)):
                line += '\t' + str(matrix[i*4 + j + len(catagrory)*len(bs)*m][w])
            output.write(line + '\n')
output.close()
