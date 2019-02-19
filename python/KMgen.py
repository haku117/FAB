import random
import datetime
import sys

MAX = 99999

def one_dp():
    data = list()
    for j in range(dim):
        data.append(random.randint(1, MAX))
    return data

def one_dpc(centroid, dist):
    data = list()
    for j in range(dim):
        data.append(max(centroid[j] + random.randint(-dist, dist), 1))
    return data

def KMgen(file_name, k, dim, num, seed):

    file = open(file_name, 'w')

    random.seed(seed)

    # generate centroids
    centroids = list()
    cluster_num = list()
    sum_num = 0
    for i in range(k):
        centroids.append(one_dp())
        c_num = random.randint(1, MAX)
        sum_num += c_num
        cluster_num.append(c_num)

    # random portion
    for i in range(k):
        cluster_num[i] = (int)(cluster_num[i] * num/sum_num)

    # generate cluster data
    dataset = list()
    tt_num = 0
    for i in range(k):
        for j in range(cluster_num[i]):
            dataset.append(one_dpc(centroids[i], MAX/100))
            tt_num += 1

    random.shuffle(dataset)

    # add some random dp if not meet num
    while tt_num < num:
        dataset.append(one_dp())
        tt_num += 1

    # save file
    for dd in dataset:
        for i in dd:
            file.write(str(i) + '\t')
        file.write('\n')

k = 4
dim = 10
num = 100

print (sys.argv)

if len(sys.argv) > 1:
    k = int(sys.argv[1])
if len(sys.argv) > 2:
    dim = int(sys.argv[2])
if len(sys.argv) > 3:
    num = int(sys.argv[3])

str_num = str(num)
if num > 1000000:
    str_num = str(num / 1000000) + 'm'
elif num > 1000:
    str_num = str(num / 1000) + 'k'

file_name = "km-" + str(k) + "-" + str(dim) + "-" + str_num + ".csv"

print("KMgen write to file " + file_name)

start = datetime.datetime.now()
KMgen(file_name, k, dim, num, 0)
end = datetime.datetime.now()

print("KMgen finish in " + str((end - start).microseconds/1000) + " ms")

# random.seed(0)
# for i in range(100):
#     print(random.randint(-100, 100))










