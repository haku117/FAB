import csv
import timeit
import sys
# import numpy as np
from Kmeans import eu_dist, load_dataset


def post_process(data_file_name, param_file_name, output_file_name, k, dim):

    s0 = timeit.default_timer()
    data_file = open(data_file_name)
    dataset = load_dataset(csv.reader(data_file), dim)
    s1 = timeit.default_timer()
    print("Finish loading dataset in %.5f s" % (s1 - s0))

    # parse parameter file
    parm_file = open(param_file_name)
    param_lines = csv.reader(parm_file)
    # top_lines(params, 2)

    output = open(output_file_name, 'w')

    start = timeit.default_timer()
    i = 0
    for param in param_lines:
        tt_loss = 0
        centroids = param2centroids(param, k, dim)
        #print(centroids)
        for dp in dataset:
            tt_loss += loss(dp, centroids)
        output.write(str(param[1]) + ',' + str(tt_loss) + '\n')

        i += 1
        # if i % 10 == 0:
        cur = timeit.default_timer()
        print("Iter %s in %.5f s" % (i, cur - start))

    data_file.close()
    parm_file.close()
    output.close()


def loss(dp, centroids):
    dp_loss = -1
    for cent in centroids:
        dist = eu_dist(cent, dp)
        if dp_loss == -1 or dist < dp_loss:
            dp_loss = dist

    return dp_loss


def param2centroids(param, k, dim):

    centroids = list()
    cent = list()

    for i in range(len(param)):
        if i < 2:
            continue # skip iter and time
        elif (i-2) % (dim + 1) == dim:
            # print(cent)
            count = float(param[i])
            for j in range(len(cent)):
                cent[j] /= count
            # print(cent)
            centroids.append(cent)
            cent = list() # reset cent
        else:
            cent.append(float(param[i]))
    return centroids


def top_lines(lines, max):
    cnt = 0
    for row in lines:
        print(row)
        cnt += 1
        if cnt >= max:
            break


if __name__ == '__main__':
    # d_file = "/Users/haku/Box Sync/results/km-4-10-1k.csv"
    # p_file = "/Users/haku/Box Sync/results/sync-1.csv"
    # o_file = "/Users/haku/Box Sync/results/sync-1.txt"

    k = 5
    dim = 20
    size = 10

    # d_file = "/Users/haku/Box Sync/dataset/cars-d.csv"
    # o_file = "/Users/haku/Box Sync/dataset/cars-rst.txt"
    # s_file = "/Users/haku/Box Sync/dataset/cars-score.txt"

    #d_file = "km-" + str(k) + "-" + str(dim) + "-" + str(size) + "k.csv"
    #o_file = "km-" + str(k) + "-" + str(dim) + "-" + str(size) + "k.txt"
    #s_file = "km-" + str(k) + "-" + str(dim) + "-" + str(size) + "k-s.txt"

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    if len(sys.argv) > 2:
        dim = int(sys.argv[2])
    if len(sys.argv) > 3:
        d_file = sys.argv[3]
        o_file = sys.argv[4]
        s_file = sys.argv[5]

    # print("Start postprocess")
    start = timeit.default_timer()
    post_process(d_file, o_file, s_file, k, dim)
    end = timeit.default_timer()

    print("Postprocess finish in " + str((end - start)) + " s")


