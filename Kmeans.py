#coding: utf-8
import numpy as np
import pandas as pd

from Cluster import Base_cluster
from Cluster import Tools

class Kmeans(Base_cluster):

    def __init__(self, n, j=0.0001, count=100):
        Base_cluster.__init__(self, n, count)
        self.J = j
        self.centre_vec = np.array([[]])

    def get_data(self, data_frame, cols):
        Base_cluster.get_data(self, data_frame, cols)

    def __random_sampling(self):
        choices = np.random.permutation(self.data_mat)[:self.N]
        self.centre_vec = choices.copy()

    def __partition(self):
        for i in self.data_frame.index:
            distances = []
            for j in xrange(0, self.centre_vec.shape[0]):
                point = self.data_mat[i, :]
                centre = self.centre_vec[j, :]
                distance = Tools.dist(centre, point)
                distances.append(distance)
            min_distance = min(distances)
            _class = distances.index(min_distance)
            self.data_frame.loc[i, "result"] = _class

    def cluster(self, train_label):
        Base_cluster.cluster(self, train_label)
        self.__random_sampling()
        num = 0
        while num < self.MAX_COUNT:
            self.__partition()
            new_centres = np.zeros((self.N, self.data_mat.shape[1]))
            for i in self.data_frame.index:
                _class = self.data_frame.loc[i, "result"]
                new_centres[_class] += self.data_mat[i]
            counts = self.data_frame[["result", "class"]].groupby(["result", ]).count()
            for _class in counts.index:
                _count = counts.loc[_class, "class"]
                new_centres[_class] /= float(_count)
            flg = True
            for i in xrange(0, self.N):
                distance = Tools.dist(self.centre_vec[i], new_centres[i])
                if distance > self.J:
                    print "error in class%d: %f"%(i, distance)
                    self.centre_vec[i] = new_centres[i]
                    flg = False
            if flg:
                break
            num += 1
        self.__partition()
        return self.data_frame


if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    data = data.sample(frac=1.0)
    data = data.reset_index(drop=True)
    kmeans = Kmeans(3)
    cols = data.columns.drop("class")
    kmeans.get_data(data, cols)
    data = kmeans.cluster("result")
    print Tools.external_value(data, "class", "result")


    