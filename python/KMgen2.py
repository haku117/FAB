import random
import timeit
import sys

MAX = 99999


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


def KMgen(file_name, k, dim, num, seed):

    file = open(file_name + ".csv", 'w')
    ref = open(file_name + "-ref.txt", 'w')

    random.seed(seed)

    # generate centroids
    s0 = timeit.default_timer()

    centroids = list()
    cluster_num = list()
    sum_num = 0
    for i in range(k):
        # centroids.append(one_dp(dim))
        centroids.append(one_dp_fix((i+1) * MAX / k, dim))
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
    dataset = list()
    tt_num = 0
    for i in range(k):
        for j in range(cluster_num[i]):
            dataset.append(one_dpc(centroids[i], MAX/50, dim))
            tt_num += 1
            if tt_num % 10000 == 0:
                print("generate %s data points" % (tt_num))

    
    s2 = timeit.default_timer()
    print("finish data generation in " + str(s2 - s0) + " s" )
    random.shuffle(dataset)
    s3 = timeit.default_timer()
    print("finish shuffle in " + str(s3 - s2) + " s" )

    # add some random dp if not meet num
    while tt_num < num:
        dataset.append(one_dp(dim))
        tt_num += 1

    # save file
    for dd in dataset:
        str_d = ""
        for i in dd:
            str_d += str(i) + ','
        file.write(str_d[:len(str_d)-1] + '\n')

    file.close()
    ref.close()


if __name__ == '__main__':
    k = 5
    dim = 20
    num = 10000

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

    file_name = "../data/km-" + str(k) + "-" + str(dim) + "-" + str_num

    print("KMgen write to file " + file_name)

    start = timeit.default_timer()
    KMgen(file_name, k, dim, num, 0)
    end = timeit.default_timer()

    print("KMgen finish in " + str((end - start)) + " s")

    # random.seed(0)
    # for i in range(100):
    #     print(random.randint(-100, 100))










