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
    linNum = 0
    for line in line_array:
        linNum += 1
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
                if len(lst) < 4:
                    print(str(linNum), lst)
                else:
                    iterNum = lst[0].split(' ')[1]
                    tt_dp += int(lst[1].split(' ')[1])
                    totalCalT += float(lst[2].split(' ')[1])
                    totalWaitT += float(lst[3].split(' ')[1])
                stat = 0

    return(iterNum, "{:.3f}".format(totalCalT/nWorker), "{:.3f}".format(totalWaitT/nWorker), tt_dp)


output_name = '../../fig/resultMatrix50-500-20k.txt'
output = open(output_name, 'w')
#output.write("what")

mode = ['dcfsb', 'dcring']
wrk = [6, 9, 16] #2, 4, 8, 16]
#bs = ['1', '2', '5', '10']#, '10', '20']
catagrory = ['iter', 'calT', 'waitT', '#dp']
# bs = [3000] #50, 100, 200, 500, 1000, 2000, 10000]
bs = [10000, 20000] #kk

dim = len(mode) * len(bs) * len(catagrory)
matrix = np.zeros((dim, 4))

titleW = ''
for i in range(1, 5):
    titleW += '\tw'+str(int(2 ** i))

dir = '/tmp/gz/log/1000-10k/'
# path = '10000'

for s in range(len(bs)):
    for m in range(len(mode)):
        for w in range(len(wrk)):
            #wNum = int(2**(w+1))
            file_name = dir + str(bs[s]) + '-0.00005/' + mode[m] + '-' + str(wrk[w]) + '-1000'
            # file_name = dir + path + '-1/' + mode[m] + '-' + str(wrk[w]) + '-' + str(bs[s])
            print(file_name)
            if os.path.isfile(file_name):
                result = loadFile(file_name, wrk[w])
                print(file_name, result)
                for c in range(len(result)):
                    #matrix[s + len(mode)*j + len(catagrory)*len(bs)*m][w] = result[j]
                    matrix[len(catagrory)*len(bs)*m + len(bs)*c + s][w] = result[c]

print(matrix)
#output.write(matrix)

for m in range(len(mode)):
    for c in range(len(catagrory)):
        title = mode[m] + '-' + catagrory[c] + titleW
        output.write(title + '\n')
        for s in range(len(bs)):
            line = str(bs[s])+'kk'
            for w in range(len(wrk)):
                line += '\t' + str(matrix[len(catagrory)*len(bs)*m + len(bs)*c + s][w])
            output.write(line + '\n')
output.close()
