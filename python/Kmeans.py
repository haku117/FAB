import csv
import timeit
import numpy as np


def load_data(data_file_name, output_file_name, k, dim, iter, bs):

    s0 = timeit.default_timer()
    data_file = open(data_file_name)
    dataset = load_dataset(csv.reader(data_file), dim)
    s1 = timeit.default_timer()
    print("Finish loading dataset in %.5f s" % (s1 - s0))

    # assignments = np.zeros(len(dataset))
    assignments = list()
    for i in range(len(dataset)):
        assignments.append(-1)

    km_train(dataset, assignments, output_file_name, k, dim, iter, bs)

    data_file.close()


def km_train(dataset, assignments, output_file_name, k, dim, iter, bs):

    output = open(output_file_name, 'w')

    # init centroids as the first 5 dp
    centroids = np.zeros((k, dim))
    c_counts = np.zeros(k)
    for i in range(k):
        centroids[i] = dataset[i]
        c_counts[i] = 1

    start = timeit.default_timer()
    # training
    index = 0
    delta = np.zeros((k, dim))
    d_counts = np.zeros(k)
    for i in range(iter):
        # E-step
        for j in range(bs):
            if index >= len(dataset):
                index = 0
            dp = dataset[index]

            rst = predict(dp, centroids, c_counts)

            if rst[0] != assignments[index]:
                delta[rst[0]] += dp
                d_counts[rst[0]] += 1
                if assignments[index] >= 0:
                    delta[assignments[index]] -= dp
                    d_counts[assignments[index]] -= 1
                assignments[index] = rst[0]

            index += 1

        # M-step
        # for i in range(k):

        # print(delta)
        centroids += delta
        c_counts += d_counts
        delta = np.zeros((k, dim))
        d_counts = np.zeros(k)

        # if i in [0, 91, 183, 280, 377, 471]:
        cur = timeit.default_timer()
        output.write("%s,%.5f" % (i, cur - start))
        for j in range(k):
            for t in range(dim):
                output.write(",%.3f" % (centroids[j][t]))#/c_counts[j]))
            output.write(",%.3f" % (c_counts[j]))
        output.write("\n")

        if i % 10 == 0:
            print("Iter %s in %.5f s" % (i, cur - start))

    output.close()


def predict(dp, centroids, c_counts):

    dp_loss, uid = -1, -1
    for i in range(k):
        dist = eu_dist(centroids[i], dp, c_counts[i])
        if dp_loss == -1 or dist < dp_loss:
            dp_loss = dist
            uid = i

    return uid, dp_loss


def eu_dist(cent, dp, c_counts = 1):
    dist = 0
    for i in range(len(dp)):
        err = cent[i]/c_counts - dp[i]
        dist += err*err
    return dist


def load_dataset(data_lines, dim):
    # restore data to dataset vector
    # get the range for normalization
    dataset = list()
    max_x = np.zeros(dim)
    min_x = np.zeros(dim)
    cnt = 0
    for row in data_lines:
        dp = list()
        i = 0
        for dd in row:
            dp.append(float(dd))
            if cnt == 0:
                max_x[i] = float(dd)
                min_x[i] = float(dd)
            elif float(dd) > max_x[i]:
                max_x[i] = float(dd)
            elif float(dd) < min_x[i]:
                min_x[i] = float(dd)
            cnt += 1
            i += 1
        dataset.append(dp)

    print(dataset[0])
    # normalization
    range_x = np.zeros(dim)
    for i in range(dim):
        range_x[i] = max_x[i] - min_x[i]

    for dp in dataset:
        for i in range(dim):
            dp[i] = 2 * (dp[i] - min_x[i]) / range_x[i] - 1

    print(dataset[0])
    return dataset



if __name__ == '__main__':
    # d_file = "/Users/haku/Box Sync/results/km-4-10-1k.csv"
    # p_file = "/Users/haku/Box Sync/results/sync-1.csv"
    # o_file = "/Users/haku/Box Sync/results/sync-1.rst"

    # d_file = "/Users/haku/Box Sync/dataset/cars-d.csv"
    # o_file = "/Users/haku/Box Sync/dataset/cars-rst.txt"

    k = 5
    dim = 20
    size = 10

    iter = 50
    bs = 1000

    d_file = "km-" + str(k) + "-" + str(dim) + "-" + str(size) + "k.csv"
    o_file = "km-" + str(k) + "-" + str(dim) + "-" + str(size) + "k.txt"

    k = 5

    # load_data(d_file, p_file, o_file, 4, 10)
    print("Start Kmeans Alg")
    start = timeit.default_timer()
    load_data(d_file, o_file, k, dim, iter, bs)
    end = timeit.default_timer()

    print("Kmeans finish in " + str((end - start)) + " s")
