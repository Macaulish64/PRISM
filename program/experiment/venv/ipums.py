# -*- coding: UTF-8 -*-
import numpy as np
import csv
# import init_censusdata
# import bi_kmeans
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA

def encode(num):
    path = 'E:\PyWorkSpace\program/ipums.la.csv'
    size = 1000000
    dimensions = 30

    dataset = []
    with open(path, 'r', encoding='UTF-8-sig') as f:
        reader = csv.reader(f)
        dataset = list(reader)

    dataset = np.array(dataset)
    dataset = dataset.astype(np.float)
    m1 = np.shape(dataset)[0]  # 数据集规模
    print(m1)
    temp_dataset = np.zeros([m1, num])
    #统一值域
    for domain in [10,20,30,40,50]:
        for j in range(num):
            tc = dataset[:, j]
            tc = tc.reshape(-1, 1)
            kmeans = KMeans(n_clusters=domain, random_state=0).fit(tc)
            print(j)
            tct = kmeans.labels_
            temp_dataset[:, j] = tct
        # 扩展维度
        pca = PCA(n_components=1)
        insert_data = pca.fit_transform(temp_dataset[:, :num])
        data = np.insert(temp_dataset, num, insert_data.T, 1)
        m2, n = data.shape

        #扩展样本数
        temp_total = []
        for i in range(size):
            temp = []
            for j in range(dimensions):
                temp.append(data[np.random.randint(0, m2 - 1)][j])
            temp_total.append(temp)
        temp_total = np.array(temp_total)
        print('The shape:', temp_total.shape)
        for j in range(49, dimensions):
            tc = temp_total[:, j]
            tc = tc.reshape(-1, 1)
            # print(tc.shape)
            kmeans = KMeans(n_clusters=domain, random_state=0).fit(tc)
            # print(j)
            tct = kmeans.labels_
            # print(temp_dataset[:,j].shape)
            temp_total[:, j] = tct
        # 输出文件
        temp_dataset = np.array(temp_total, dtype=int)
        name = 'ipums_'+str(domain) +'-' + str(dimensions) + '-1M.txt'
        np.savetxt(name, temp_dataset, delimiter=",", fmt="%d")
        # print (dataset_01[1])
        temp_dataset = np.zeros([m1, num])

if __name__ == '__main__':
    encode(30)

