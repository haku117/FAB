import os.path
import numpy as np
import math


def loadFile(file_name, nWorker):

    file = open(file_name, 'r')
    line_array = file.readlines()

    totalCalT = 0
    totalWaitT = 0
    iterNum = 0
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

    return(iterNum, "{:.3f}".format(totalCalT/nWorker), "{:.3f}".format(totalWaitT/nWorker))


output_name = '../../boxCOM/figure/resultMatrix.txt'
output = open(output_name, 'w')
#output.write("what")

mode = ['fsb', 'dcfsb']
wrk = [2, 4, 8, 16]
bs = ['1', '5', '10', '20']

matrix = np.zeros((24, 4))

titleW = ''
for i in range(1, 5):
    titleW += '\tw'+str(int(2 ** i))

dir = '../build/log/10-1m/'
path = '000-0.01/'

for s in range(4):
    for m in range(2):
        for w in range(4):
            wNum = int(2**(w+1))
            file_name = dir + bs[s] + path + mode[m] + '-' + str(wNum)
            if os.path.isfile(file_name):
                result = loadFile(file_name, wNum)
                print(file_name, result)
                for j in range(3):
                    matrix[s + 4*j + 12*m][w] = result[j]

print(matrix)
#output.write(matrix)

catagrory = ['iter', 'calT', 'waitT']
for m in range(2):
    for i in range(3):
        title = mode[m] + '-' + catagrory[i] + titleW
        output.write(title + '\n')
        for j in range(4):
            line = bs[j]+'k'
            for k in range(4):
                line += '\t' + str(matrix[i*4 + j + 12*m][k])
            output.write(line + '\n')
output.close()
