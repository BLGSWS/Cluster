#coding: utf-8

import numpy as np
from numpy import linalg as LA
import pandas as pd

class Base_cluster(object):

    def __init__(self, n, count):
        self.N = n
        self.MAX_COUNT = count
        self.data_frame = pd.DataFrame()
        self.data_mat = np.array([[]])
   
    def get_data(self, data_frame, cols):
        self.data_frame = data_frame
        for col in cols:
            if col not in self.data_frame.columns:
                raise Exception()
        self.data_mat = np.array(data_frame[cols].values.astype("float64"))
        assert data_frame.shape[0] >= self.N
   
    def cluster(self, train_label):
        '''
        :param train_label: 聚类结果标签
        '''
        if self.data_frame.shape[0] == 0:
            raise Exception()
        self.data_frame[train_label] = -1

class Tools(object):

    @staticmethod
    def get_map(data_frame, reference_label, train_label, n, sample_num=10):
        '''
        :param data_frame: 带有聚类标记的数据集
        :param label: 参考标记标签
        :param train_label: 聚类结果标签
        :param n: 聚类族数
        :
        '''
        assert(reference_label in data_frame.columns)
        assert(train_label in data_frame.columns)
        assert(n*sample_num <= data_frame.shape[0])
        count_dict = {}
        map_dict = {}
        for i in xrange(n*sample_num):
            instance = "%s+%s"%(data_frame.loc[reference_label, i], data_frame.loc[train_label, i])
            if instance not in count_dict.keys():
                count_dict[instance] = 1
            else:
                count_dict[instance] += 1
        #选出前N
        for _ in xrange(n):
            max_value = 0
            max_instance_key = ""
            for key in count_dict:
                if int(count_dict[key]) > max_value:
                    max_value = int(count_dict[key])
                    max_instance_key = key
            max_instance = max_instance_key.split("+")
            map_dict[max_instance[0]] = max_instance[1]
            del count_dict[max_instance_key]
        return map_dict

    @staticmethod
    def get_accuracy(data_frame, label, train_label, class_map):
        assert(label in data_frame.columns)
        assert(train_label in data_frame.columns)
        count = 0
        for i in data_frame.index:
            if class_map[data_frame.loc[i, train_label]] == data_frame.loc[i, label]:
                count += 1
        accuracy = float(count)/data_frame.shape[0]
        return accuracy

    @staticmethod
    def __statistic(data_frame, label, train_label):
        SS = 0
        SD = 0
        DS = 0
        DD = 0
        for i in xrange(0, data_frame.shape[0]):
            for j in xrange(i + 1, data_frame.shape[0]):
                if data_frame.loc[i, label] == data_frame.loc[j, label]:
                    if data_frame.loc[i, train_label] == data_frame.loc[j, train_label]:
                        SS += 1
                    else:
                        SD += 1
                else:
                    if data_frame.loc[i, train_label] == data_frame.loc[j, train_label]:
                        DS += 1
                    else:
                        DD += 1
        return float(SS), float(SD), float(DS), float(DD)

    @staticmethod
    def external_value(data_frame, reference_label, train_label, value_type = "Jaccard"):
        m = data_frame.shape[0]
        SS, SD, DS, DD = Tools.__statistic(data_frame, reference_label, train_label)
        if value_type == "Jaccard":
            return SS/(SS + SD + DS)
        elif value_type == "FMI":
            return (SS*SS/(SS + SD)/(SS + DS))**0.5
        elif value_type == "RI":
            return 2*(SS + DD)/m*(m - 1)
        else:
            raise Exception()
    
    @staticmethod
    def internal_value(data_frame, train_label, value_type = "DB"):
        pass

    @staticmethod
    def dist(x, y):
        vec = x - y
        distance = LA.norm(vec)
        return distance

if __name__ == "__main__":
    x = np.array([2, 1, 2])
    y = np.array([0, 0, 0])
    print Tools.dist(x, y)