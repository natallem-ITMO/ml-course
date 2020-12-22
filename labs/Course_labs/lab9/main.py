import pandas as pd
from math import sqrt

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from random import randrange

def load_dataset():
    str = "dataset/wine.csv"
    return pd.read_csv(str)


def dataset_minmax(dataset):
    minmax = [[x, x] for x in dataset[0]]
    for row in dataset:
        for idx, x in enumerate(row):
            minmax[idx][0] = min(minmax[idx][0], x)
            minmax[idx][1] = max(minmax[idx][1], x)
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if (minmax[i][1] - minmax[i][0] == 0):
                row[i] = 1
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def calc_dist(row1, row2):
    sum = 0
    for (i, j) in zip(row1, row2):
        sum += (i - j) ** 2
    return sqrt(sum)


def find_neighbors(dataset):
    result = []
    distances = []
    for i in range(len(dataset)):
        cur_list = []
        cur_dist = []
        for j in range(len(dataset)):
            if j == i:
                cur_dist.append(0)
                continue
            dist = calc_dist(dataset[i], dataset[j])
            cur_dist.append(dist)
            if (dist < eps):
                cur_list.append(j)
        result.append(cur_list)
        distances.append(cur_dist)
    return (result, distances)


def calc_DBSCAN(neighbors, dataset):
    num_clust = 0
    set_not_visited = set([i for i in range(len(dataset))])
    noise = set()
    clust = [-1 for i in range(len(dataset))]  # (el_num , pair(num_clust, is_edge))
    while set_not_visited:
        cur = set_not_visited.pop()
        if len(neighbors[cur]) < m:
            noise.add(cur)
            continue
        clust[cur] = num_clust
        cur_clust = set(neighbors[cur])
        while cur_clust:
            cur_clust_el = cur_clust.pop()
            if (cur_clust_el in noise) or (cur_clust_el in set_not_visited):
                if (cur_clust_el in noise):
                    noise.remove(cur_clust_el)
                if (cur_clust_el in set_not_visited):
                    set_not_visited.remove(cur_clust_el)
                if (len(neighbors[cur_clust_el]) < m):
                    clust[cur_clust_el] = num_clust
                else:
                    for x in neighbors[cur_clust_el]:
                        cur_clust.add(x)
                    clust[cur_clust_el] = num_clust
        num_clust += 1
    return (num_clust, clust)


def calc_kmeans(dataset, clust_num):
    centers = [dataset[randrange(0, len(dataset))].copy() for i in range(clust_num)]
    result = [0 for i in range(len(dataset))]
    while (True):
        clusters = [[] for i in range(clust_num)]
        for (ind, x) in enumerate(dataset):
            min_class = 0
            min_dist = calc_dist(centers[0], x)
            for (ind_cl, cent) in enumerate(centers):
                cur_dist = calc_dist(cent, x)
                if (min_dist > cur_dist):
                    min_class = ind_cl
                    min_dist = cur_dist
            clusters[min_class].append(x)
            result[ind] = min_class
        new_centers = list()
        for i in range(clust_num):
            if (len(clusters[i]) == 0):
                new_centers.append(dataset[randrange(0, len(dataset))].copy())
                break
            summa = clusters[i][0]
            for j in range(1, len(clusters[i])):
                summa = summa + clusters[i][j]
            new_centers.append([x / len(clusters[i]) for x in summa])
        if (new_centers != centers):
            break
        centers = new_centers
    return result


def calc_max_kmeans(dataset, num_clusters, classes):
    clusters = calc_kmeans(dataset, num_clusters)
    measure = calc_external_measure(clusters, dataset, classes)
    repetitions = 20
    for i in range(repetitions):
        cur_clusters = calc_kmeans(dataset, num_clusters)
        sett = set(cur_clusters)
        if (len(sett) != num_clusters):
            i -= 1
            continue
        cur_measure = calc_external_measure(cur_clusters, dataset, classes)
        if (cur_measure > measure):
            clusters = cur_clusters
            measure = cur_measure
    return (clusters, measure)


def calc_silhouette(clusters, num_clusters, dataset):  # Dunn Index
    clusters_list = [list() for i in range(num_clusters)]

    for (ind, clust) in enumerate(clusters):
        clusters_list[clust].append(ind)
    summa = 0
    for clust in range(num_clusters):
        for ind in clusters_list[clust]:
            b = -1
            a = sum([calc_dist(dataset[ind], dataset[i]) for i in clusters_list[clust]]) / len(clusters_list[clust])
            for clustA in range(num_clusters):
                if clustA == clust: continue
                cur_res = sum([calc_dist(dataset[ind], dataset[i]) for i in clusters_list[clustA]]) / len(
                    clusters_list[clustA])
                if (b == -1 or b < cur_res):
                    b = cur_res
            summa += (b - a) / max(a, b)
    return summa / len(dataset)


def calc_external_measure(clust, dataset, classes):  # Rand index
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            in_one_class = classes[i] == classes[j]
            in_one_clust = clust[i] == clust[j]
            if in_one_class and clust[i] == -1:
                in_one_class = False
            if in_one_class and in_one_clust:
                TP += 1
            if in_one_class and not in_one_clust:
                FP += 1
            if not in_one_class and in_one_clust:
                TN += 1
            if not in_one_class and not in_one_clust:
                FN += 1
    return (TP + FN) / (TP + FN + FP + TN)


def draw_plot(dataset, classes, clusters):
    pca = PCA(n_components=2)
    pca.fit(dataset)
    res = pca.transform(dataset)
    x = list()
    y = list()
    for i in res:
        x.append(i[0])
        y.append(i[1])

    plt.scatter(x, y, c=classes, alpha=0.3,
                cmap='viridis')
    plt.colorbar()
    plt.savefig("origin.png", bbox_inches='tight')
    plt.close()
    plt.scatter(x, y, c=clusters, alpha=0.3,
                cmap='viridis')
    plt.colorbar()
    plt.savefig("cluster.png", bbox_inches='tight')
    plt.close()


def drow_griphic(dataset, classes):
    max_clusters = 14
    x = list()

    y_ext = list()
    y_int = list()
    for i in range(1, max_clusters):
        x.append(i)
        [clusters, ext_measure] = calc_max_kmeans(dataset, i, classes)
        y_ext.append(ext_measure)
        y_int.append(calc_silhouette(clusters, i, dataset))

    plt.plot(x, y_int, '.-y', alpha=0.6, label="internal measure", lw=3)
    plt.xscale("linear")
    plt.xlabel("number of clusters")
    plt.ylabel("internal measure")
    plt.legend()
    plt.savefig("internal_measure.png", bbox_inches='tight')
    plt.close()
    plt.plot(x, y_ext, '.-b', alpha=0.6, label="external measure", lw=3)
    plt.xscale("linear")
    plt.xlabel("number of clusters")
    plt.ylabel("external measure")
    plt.legend()
    plt.savefig("external_measure.png", bbox_inches='tight')
    plt.close()


data_frame = load_dataset()
dataset = data_frame.loc[:, data_frame.columns != 'class'].values.tolist().copy()
classes = data_frame.loc[:, data_frame.columns == 'class'].values.tolist().copy()
min_max = dataset_minmax(dataset)
normalize(dataset, min_max)

num_clusters = 3
[clusters, ext_measure] = calc_max_kmeans(dataset, num_clusters, classes)

draw_plot(dataset, classes, clusters)
drow_griphic(dataset, classes)
