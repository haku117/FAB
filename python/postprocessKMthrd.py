import csv
import timeit
import time
import sys
import numpy as np
import threading


class ppThread(threading.Thread):

    def __init__(self, threadID, dataset, centroids=0, k=0, dim=0, o_file="", s_file="", threads=[]):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # self.name = name
        self.dt = dataset
        self.ct = centroids
        self.needTrain = False
        self.k = k
        self.dim = dim
        self.p_file = p_file
        self.r_file = r_file
        self.threads = threads

        global thrd_done

    def resetCentroids(self, centroids):
        self.ct = centroids
        # print("Thread=" + str(self.threadID) + " centroids reset ")
        self.needTrain = True

    def run(self):
        print("Starting " + self.name)
        # cal_loss(self.dt, self.ct, self.threadID)
        # compute_sum(self.threadID, int(self.threadID), 2)

        if self.threadID == 0: # Master
            master_param(self.p_file, self.r_file, self.k, self.dim, self.threads)
        else: #workers
            cnt = 0
            while thrd_done >= 0:
                time.sleep(1)
                # print("Thread=" + str(self.threadID) + " waiting " + str(cnt) + " glb_done " + str(thrd_done))
                cnt += 1
                if self.needTrain:
                    # glb_test(self.threadID, 1)
                    cal_loss(self.dt, self.ct, self.threadID)
                    self.needTrain = False


def glb_test(threadID, delay):
    global thrd_done, threadLock

    print("Thread=" + str(threadID) + " running " + str(thrd_done))
    # time.sleep(delay)
    threadLock.acquire()
    thrd_done += 1
    threadLock.release()
    print("thrd_done %s from Thread-%s in: %s" % (thrd_done, threadID, time.ctime(time.time())))


def cal_loss(dt, ct, threadID):
    global threadLock, tt_loss, thrd_done

    t1 = timeit.default_timer()
    local_loss = 0
    for dp in dt:
        local_loss += loss(dp, ct)

    # print("Thread=" + str(threadID) + " requiring lock " + str(thrd_done))
    t2 = timeit.default_timer()
    threadLock.acquire()
    tt_loss += local_loss
    thrd_done += 1
    threadLock.release()

    if need_log:
        print("sum %s %s from Thread-%s with dt %s in: %s" % (tt_loss, thrd_done, threadID, len(dt), (t2-t1)))


def post_process(data_file_name, param_file_name, output_file_name,
                 k, d_size, dim, thrd_num, need_log):

    global tt_loss, thrd_done

    s0 = timeit.default_timer()
    data_file = open(data_file_name)
    dataset_thrd = load_dataset(csv.reader(data_file), dim, thrd_num, d_size)
    s1 = timeit.default_timer()
    print("Finish loading dataset in %.5f s" % (s1 - s0))

    threads = list()
    # start master thread
    threads.append(ppThread(0, 0, 0, k, dim, param_file_name, output_file_name, threads))
    threads[0].start()

    for tr in range(thrd_num):
        threads.append(ppThread(tr+1, dataset_thrd[tr]))
        threads[tr+1].start()

    for t in threads:
        t.join(1)

    # while threading.activeCount() > 0:
    #     print(threading.enumerate())
    #     for t in threads:
    #         if t.is_alive():
    #             print(t.name, "is alive")
    #             t.join(1)
    #         else:
    #             print(t.name, "is teminated")

    data_file.close()


def master_param(param_file_name, output_file_name, k, dim, threads):

    global thrd_done, threadLock

    # for tt in range(1, len(threads)):
    #     threads[tt].join(1)

    parm_file = open(param_file_name)
    param_lines = csv.reader(parm_file)
    output = open(output_file_name, 'w')

    last_c = 0
    last_loss = 0
    p_iter = 0

    for param in param_lines:
        tt_loss = 0

        # if p_iter > 5:
        #     break

        t1 = timeit.default_timer()
        print("Start process param " + str(p_iter))
        centroids = param2centroids(param, k, dim)
        if last_c == 0:
            last_c = centroids
        elif eu_dist_c(centroids, last_c) < 1e-6:
            last_c = centroids
            p_iter += 1
            # if i % 10 == 0:
            cur = timeit.default_timer()
            if need_log > 0:
                print("Iter %s in %.5f s with diff %s" % (p_iter, cur - t1, eu_dist_c(centroids, last_c)))
            output.write(str(param[1]) + ',' + str(last_loss) + '\n')
            continue

        for t in range(thrd_num):
            # print("Go reset centroids for " + str(t+1))
            threads[t+1].resetCentroids(centroids)

        while(thrd_done != thrd_num):
            time.sleep(1)
            print("main sleep")
            continue

        thrd_done = 0

        cur = timeit.default_timer()
        if need_log > 0:
            print("Iter %s in %.5f s with diff %s" % (p_iter, cur - t1, eu_dist_c(centroids, last_c)))
        output.write(str(param[1]) + ',' + str(tt_loss) + '\n')

        p_iter += 1
        # if i % 10 == 0:
        cur = timeit.default_timer()
        if need_log > 0:
            print("Iter %s in %.5f s with diff %s" % (p_iter, cur - t1, eu_dist_c(centroids, last_c)))
        last_c = centroids
        last_loss = tt_loss

    threadLock.acquire()
    thrd_done = -1
    threadLock.release()
    print("Finish master thread", p_iter)

    # while threading.activeCount() > 1:
    print(threading.enumerate())
    #
    #     for tt in range(1, len(threads)):
    #         if threads[tt].is_alive():
    #             print(threads[tt].name, "is alive")
    #             threads[tt].join(1)
            # else:
            #     print(t.name, "is teminated")

    parm_file.close()
    output.close()


def eu_dist(cent, dp, c_counts = 1):
    dist = 0
    for i in range(len(dp)):
        err = cent[i]/c_counts - dp[i]
        dist += err*err
    return dist


def eu_dist_c(cent, last_c):
    dist = 0
    for i in range(len(cent)):
        for j in range(len(cent[i])):
            err = cent[i][j] - last_c[i][j]
            dist += err*err
    return dist


def loss(dp, centroids):
    dp_loss = -1
    for cent in centroids:
        dist = eu_dist(cent, dp)
        if dp_loss == -1 or dist < dp_loss:
            dp_loss = dist

    return dp_loss


def load_dataset(data_lines, dim, thrd_num, d_size):
    # restore data to dataset vector
    data_thrd = list()
    data_part_num = 1 + d_size / thrd_num

    dataset = list()
    cnt_row = 0
    for row in data_lines:
        dp = list()
        for data_dim in row:
            dp.append(float(data_dim))
        cnt_row += 1
        dataset.append(dp)
        if cnt_row % data_part_num == 0:
            data_thrd.append(dataset)
            # print("dp size: " + str(len(dataset)))
            dataset = list()

    if len(dataset) != 0:
        data_thrd.append(dataset)

    print(" data partition num: " + str(len(data_thrd)) + " to " + str(thrd_num))

    return data_thrd


def load_dataset_norm(data_lines, dim, thrd_num, d_size):
    # restore data to dataset vector
    # get the range for normalization
    data_thrd = list()
    data_part_num = 1 + d_size / thrd_num

    dataset = list()
    max_x = np.zeros(dim)
    min_x = np.zeros(dim)
    cnt_row = 0
    for row in data_lines:
        dp = list()
        i_dim = 0  # for dimension
        for data_dim in row:
            dp.append(float(data_dim))
            if cnt_row == 0:
                max_x[i_dim] = float(data_dim)
                min_x[i_dim] = float(data_dim)
            elif float(data_dim) > max_x[i_dim]:
                max_x[i_dim] = float(data_dim)
            elif float(data_dim) < min_x[i_dim]:
                min_x[i_dim] = float(data_dim)
            i_dim += 1
        cnt_row += 1
        dataset.append(dp)
        if cnt_row % data_part_num == 0:
            data_thrd.append(dataset)
            dataset = list()

    if len(dataset) != 0:
        data_thrd.append(dataset)

    print(" data partition num: " + str(len(data_thrd)) + " to " + str(thrd_num))

    # normalization
    range_x = np.zeros(dim)
    for i in range(dim):
        range_x[i] = max_x[i] - min_x[i]

    for dataset in data_thrd:
        for dp in dataset:
            for i in range(dim):
                dp[i] = 2 * (dp[i] - min_x[i]) / range_x[i] - 1

    # print(dataset[0])
    return data_thrd


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


threadLock = threading.Lock()
tt_loss = 0
total_sum = 0
thrd_done = 0

if __name__ == '__main__':
    # d_file = "/Users/haku/Box Sync/results/km-4-10-1k.csv"
    # p_file = "/Users/haku/Box Sync/results/sync-1.csv"
    # o_file = "/Users/haku/Box Sync/results/sync-1.txt"

    k = 5
    dim = 50
    size = 10
    thrd_num = 1
    need_log = 1

    # d_file = "/Users/haku/Box Sync/dataset/cars-d.csv"
    # o_file = "/Users/haku/Box Sync/dataset/cars-rst.txt"
    # s_file = "/Users/haku/Box Sync/dataset/cars-score.txt"

    flder = "/Users/haku/PycharmProjects/KMgen/"

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    if len(sys.argv) > 2:
        dim = int(sys.argv[2])
    if len(sys.argv) > 3:
        size = int(sys.argv[3])
    if len(sys.argv) > 4:
        thrd_num = int(sys.argv[4])
    if len(sys.argv) > 5:
        need_log = int(sys.argv[5])
    if len(sys.argv) > 6:
        d_file = sys.argv[6]
    if len(sys.argv) > 7:
        p_file = sys.argv[7]
    if len(sys.argv) > 8:
        r_file = sys.argv[8]

    # print("Start postprocess")
    start = timeit.default_timer()
    post_process(d_file, p_file, r_file, k, size*1000, dim, thrd_num, need_log)
    end = timeit.default_timer()

    print("Postprocess finish in " + str((end - start)) + " s")


