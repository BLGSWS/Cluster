#coding: utf-8
import numpy as np
import pandas as pd

def read(filepath):
    data = []
    with file(filepath, "r") as myfile:
        while 1:
            line = myfile.readline()
            line = line.replace("\n", "")
            if not line:
                break
            attributes = line.split(",")
            data.append(attributes)
        mat = np.array(data)
        frame = pd.DataFrame(data=mat,
                             columns=["Sepal_length",
                                      "Speal_width",
                                      "Petal_lenth",
                                      "Petal_width",
                                      "test_label"])
        frame = frame.sample(frac=1.0).reset_index(drop=True)
        frame.to_csv("Iris.csv")
        return frame

class FCM(object):

    def __init__(self, m=4, n=3, j=0.0001, count=100):
        self.M = m
        self.N = n
        self.J = j
        self.COUNT = count
        self.data_frame = pd.DataFrame()
        self.data_mat = [[]]
        self.map_dict = {}

    def get_data(self, data_frame, cols):
        self.data_frame = data_frame
        self.data_mat = np.array(data_frame[cols]).astype("float64")

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

    def __select(self, u_mat):
        predict = []
        for i in xrange(u_mat.shape[1]):
            kind = np.argmax(np.transpose(u_mat[:, i]))
            predict.append(kind)
        return predict

    def kind_map(self, test_label):
        count_dict = {}
        map_dict = {}
        for i in xrange(self.N*10):
            instance = "%s_%s"%(self.data_frame[test_label][i], self.data_frame.train_label[i])
            if instance not in count_dict.keys():
                count_dict[instance] = 1
            else:
                count_dict[instance] += 1
        #选出前N
        for _ in xrange(self.N):
            max_value = 0
            max_instance_key = ""
            for key in count_dict:
                if int(count_dict[key]) > max_value:
                    max_value = int(count_dict[key])
                    max_instance_key = key
            max_instance = max_instance_key.split("_")
            map_dict[max_instance[0]] = max_instance[1]
            del count_dict[max_instance_key]
        count = 0
        for i in xrange(self.data_frame.shape[0]):
            if int(map_dict[self.data_frame.test_label[i]]) == int(self.data_frame.train_label[i]):
                count += 1
        print "accuracy: %f"%(float(count)/self.data_frame.shape[0])
        self.map_dict = map_dict
        return map_dict

    def cluster(self):
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
        for _ in xrange(self.COUNT):
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
        predict = self.__select(u_mat)
        self.data_frame["train_label"] = pd.Series(predict)


