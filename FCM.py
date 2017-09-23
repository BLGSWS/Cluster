#coding: utf-8
import pandas as pd
import numpy as np

from Cluster import Base_cluster
from Cluster import Tools

class FCM(Base_cluster):

    def __init__(self, n, m=4, j=0.0001, count=100):
        Base_cluster.__init__(self, n, count)
        self.M = m
        self.J = j
    
    def get_data(self, data_frame, cols):
        Base_cluster.get_data(self, data_frame, cols)

    def __distant(self, x, y):
        return np.sum((x-y)**2)**0.5

    def __c_iterator(self, x_mat, u_mat, c_mat):
        for i in xrange(self.N):
            c_mat[i, :] = np.dot(u_mat[i, :]**self.M, x_mat)/np.sum(u_mat[i, :]**self.M)
        return c_mat

    def __u_iterator(self, x_mat, u_mat, c_mat):
        for i in xrange(u_mat.shape[0]):
            for j in xrange(u_mat.shape[1]):
                den = 0.0
                for k in xrange(self.N):
                    den += (self.__distant(x_mat[j, :], c_mat[i, :])/\
                    self.__distant(x_mat[j, :], c_mat[k, :]))**(2.0/(self.M-1))
                u_mat[i, j] = 1.0/den
        return u_mat

    def __error(self, x_mat, u_mat, c_mat):
        err = 0.0
        for i in xrange(u_mat.shape[0]):
            for j in xrange(u_mat.shape[1]):
                err += u_mat[i, j]**self.M*self.__distant(x_mat[j, :], c_mat[i, :])**2.0
        return err

    def cluster(self, train_label="result"):
        Base_cluster.cluster(self, train_label)
        #随机生成隶属矩阵u_mat
        u_mat = np.random.random((self.N, self.data_mat.shape[0]))
        for i in xrange(u_mat.shape[1]):
            col_sum = np.sum(u_mat[:, i])
            for j in xrange(u_mat.shape[0]):
                u_mat[j, i] = u_mat[j, i]/col_sum
        #初始化聚类中心
        c_mat = np.zeros((self.N, self.data_mat.shape[1]))
        error1 = 1.0
        #迭代
        count = 0
        for _ in xrange(self.MAX_COUNT):
            c_mat = self.__c_iterator(self.data_mat, u_mat, c_mat)
            u_mat = self.__u_iterator(self.data_mat, u_mat, c_mat)
            error2 = self.__error(self.data_mat, u_mat, c_mat)
            print "error: %f"%error2
            if abs(error2-error1) < self.J:
                count += 1
                if count > 3:
                    break
            error1 = error2
        #分类
        for i in xrange(u_mat.shape[1]):
            _class = np.argmax(np.transpose(u_mat[:, i]))
            self.data_frame.loc[i, train_label] = _class
        return self.data_frame

if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    fcm = FCM(3)
    cols = data.columns.drop("class")
    fcm.get_data(data, cols)
    fcm.cluster("result")
    print Tools.external_value(data, "result", "class")


