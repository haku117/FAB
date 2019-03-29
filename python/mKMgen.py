import random
import timeit
import sys
import numpy as np
import csv

MAX = 10


def one_dp(dim):
    data = list()
    for j in range(dim):
        data.append(random.randint(1, MAX))
    return data


def one_dp_fix(ff, dim):
    data = list()
    for j in range(dim):
        data.append(random.randint(ff - MAX/10, ff + MAX/ 10))
    return data


def one_dpc(centroid, dist, dim):
    data = list()
    for j in range(dim):
        data.append(max(centroid[j] + random.randint(-dist, dist), 1))
    return data


def KMgenMatrix(file_name, dim, num, seed):

    ff = open(file_name + ".csv", 'w')

    random.seed(seed)

    s0 = timeit.default_timer()
    for j in range(num):
        dd = one_dp(dim)
        str_d = ""
        for i in dd:
            str_d += str(i) + ','
        ff.write(str_d[:len(str_d)-1] + '\n')
        if j%num == 1000:
            s2 = timeit.default_timer()
            print("generate", j, "lines in", str(s2 - s0), " s")

    s2 = timeit.default_timer()
    print("finish data generation in " + str(s2 - s0) + " s")
    ff.close()

def KMgenMatrix2(file_name, dim, num):

    ff = open(file_name + ".csv", 'w')

    s0 = timeit.default_timer()
    for j in range(num):
        dd = np.random.randint(1, high=MAX, size=dim)
        #dd2 = np.random.rand(dim)
        s3 = timeit.default_timer()
        str_d = ""
        for i in range(len(dd)-1):
            str_d += str(dd[i]) + ','
            #ff.write(str(dd[i])+',')
        #s4 = timeit.default_timer()
        #str_d2 = ""
        #for i in range(len(dd2)-1):
        #    str_d2 += str(dd2[i]*10)[2] + ','
            #ff.write(str(dd[i])+',')
        s5 = timeit.default_timer()

        ff.write(str_d + str(dd[i])+'\n')

        if j%200 == 199:
            s2 = timeit.default_timer()
            print("generate", j, "lines in", str(s2 - s0), "s with",(s5-s3),(s2-s5))

    s2 = timeit.default_timer()
    print("finish data generation in " + str(s2 - s0) + " s")
    ff.close()


def KMgenMatrix3(file_name, dim, num):
    with open(file_name+'.csv', 'wb') as ff:
        lw = csv.writer(ff, delimiter=',')

        s0 = timeit.default_timer()
        for j in range(num):
            dd = np.random.randint(1, high=MAX, size=dim)
            s5 = timeit.default_timer()
            lw.writerow(dd)
            if j%200 == 199:
                s2 = timeit.default_timer()
                print("generate", j, "lines in", str(s2 - s0), "s with",(s2-s5))

        s2 = timeit.default_timer()
        print("finish data generation in " + str(s2 - s0) + " s")


def KMgen(file_name, k, dim, num, seed):

    ref = open(file_name + "-ref.txt", 'w')

    random.seed(seed)

    # generate centroids
    s0 = timeit.default_timer()

    centroids = list()
    cluster_num = list()
    sum_num = 0
    for i in range(k):
        centroids.append(one_dp(dim))
        #centroids.append(one_dp_fix((i+1) * MAX / k, dim))
        c_num = random.randint(1, MAX)
        sum_num += c_num
        cluster_num.append(c_num)

    # random portion
    for i in range(k):
        cluster_num[i] = int(cluster_num[i] * num/sum_num)
        str_d = ""
        for j in range(len(centroids[i])):
            str_d += str(centroids[i][j]) + ','
        ref.write(str_d[:len(str_d)-1] + "," + str(cluster_num[i]) + '\n')

    # generate cluster data
    dataset_all = list()
    dataset = list()
    tt_num = 0
    remain = 0
    for nn in cluster_num:
        remain += nn
    print("remain ", remain)

    while remain > 0:
        ck = random.randint(0, k-1)
        if cluster_num[ck] == 0:
            continue
        dataset.append(one_dpc(centroids[ck], MAX/10, dim))
        tt_num += 1
        remain -= 1
        cluster_num[ck] -= 1
        if tt_num % 1000 == 0:
            print("generate %s data points" % (tt_num))
            # save2file(file_name, dataset)
            dataset_all.append(dataset)
            dataset = list()

    # random.shuffle(dataset)
    # s3 = timeit.default_timer()
    # print("finish shuffle in " + str(s3 - s2) + " s" )

    # add some random dp if not meet num
    print("tt_num ", tt_num)
    while tt_num < num:
        dataset.append(one_dp(dim))
        tt_num += 1

    print("tt_num ", tt_num)
    print("data remain ", len(dataset))
    dataset_all.append(dataset)

    normalization(dataset_all, dim)

    save2file(file_name, dataset_all)

    s2 = timeit.default_timer()
    print("finish data generation in " + str(s2 - s0) + " s")

    ref.close()


def normalization(dataset_all, dim):
    max_x = np.zeros(dim)
    min_x = np.zeros(dim)
    cnt = 0
    for ds in dataset_all:
        for row in ds:
            i = 0
            for dd in row:
                if cnt == 0:
                    max_x[i] = float(dd)
                    min_x[i] = float(dd)
                elif float(dd) > max_x[i]:
                    max_x[i] = float(dd)
                elif float(dd) < min_x[i]:
                    min_x[i] = float(dd)
                cnt += 1
                i += 1

    # normalization
    range_x = np.zeros(dim)
    for i in range(dim):
        range_x[i] = max_x[i] - min_x[i]

    for ds in dataset_all:
        for dp in ds:
            for i in range(dim):
                dp[i] = 2 * (dp[i] - min_x[i]) / range_x[i] - 1


def save2file(file_name, dataset_all):
    file = open(file_name + ".csv", 'a')
    # save file
    for dataset in dataset_all:
        for dd in dataset:
            str_d = ""
            for i in dd:
                str_d += str(i) + ','
            file.write(str_d[:len(str_d)-1] + '\n')

    file.close()


if __name__ == '__main__':
    k = 5
    dim = 50
    num = 10000

    print (sys.argv)

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    if len(sys.argv) > 2:
        dim = int(sys.argv[2])
    if len(sys.argv) > 3:
        num = int(sys.argv[3])

    str_num = str(num)
    if num >= 1000000:
        str_num = str(num / 1000000) + 'm'
    elif num >= 1000:
        str_num = str(num / 1000) + 'k'

    file_name = "../data/nmf-" + str(dim) + "-" + str_num

    print("KMgen write to file " + file_name)

    start = timeit.default_timer()
    if k > 0:
        KMgenMatrix(file_name, dim, num, 0)
    else:
        KMgenMatrix3(file_name, dim, num)
    end = timeit.default_timer()

    print("KMgen finish in " + str((end - start)) + " s")

