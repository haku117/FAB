import os.path
import numpy as np
import math


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

    return(iterNum, "{:.3f}".format(totalCalT/nWorker), "{:.3f}".format(totalWaitT/nWorker))


mode = ['fsb', 'dcfsb']
wrk = [2, 4, 8, 16]
bs = ['1k', '5k', '10k', '20k']

matrix = np.zeros((12, 4))

titleW = ''
for i in range(1, 5):
    titleW += '\tw'+str(int(2 ** i))

dir = '../build/log/10-1m/'
path = '000-0.01/'

for s in range(4):
    for m in mode:
        for w in wrk:
            file_name = m + '-' + str(w)
            if os.path.isfile(file_name):
                result = loadFile(file_name, w)
                print(file_name, result)
                for j in range(3):
                    matrix[s + 4*j][int(math.log(w))] = result[j]

print(titleW)
output_name = 'test'
output = open(output_name, 'w')
#output.write(matrix)

catagrory = ['iter', 'calT', 'waitT']
for i in range(3):
    title = 'dcfsb-' + catagrory[i] + titleW
    output.write(title + '\n')
    for j in range(4):
        line = bs[j]
        for k in range(4):
            line += '\t' + str(matrix[i*4 + j][k])
        output.write(line + '\n')
output.close()
