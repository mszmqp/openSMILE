#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: mengshuai(mengshuai@100tal.com)
Date:    2019/1/26 15:30
"""
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import os
import random
from sklearn import svm, preprocessing
from sklearn.preprocessing import label_binarize, minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion

__version__ = 1.0

class Model(object):
    name = "model"
    # 模型参数
    param = {}
    description = "基类"

    def __init__(self, x_all=None, y_all=None, train_x=None, train_y=None, test_x=None, test_y=None):
        if x_all is not None and y_all is not None:
            self.x_all = x_all
            self.y_all = y_all
            self.split_train_test()
        else:
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

        self.train_x = np.array(self.train_x).astype('float64')
        self.train_y = np.array(self.train_y).astype('float64')
        self.test_x = np.array(self.test_x).astype('float64')
        self.test_y = np.array(self.test_y).astype('float64')

    def split_train_test(self):
        self.train_x, self.test_x, self.train_y, self.test_y = [], [], [], []
        label_set = set(self.y_all)
        # 多类别数据拆分
        for label in label_set:
            index_temp = [i for i in range(len(self.y_all)) if self.y_all[i] == label]
            self.x_all_temp = [self.x_all[item] for item in index_temp]
            self.y_all_temp = [self.y_all[item] for item in index_temp]
            self.train_x_temp, self.test_x_temp, self.train_y_temp, self.test_y_temp = train_test_split(self.x_all_temp, self.y_all_temp, test_size=0.2, random_state=1)
            self.train_x.extend(self.train_x_temp)
            self.test_x.extend(self.test_x_temp)
            self.train_y.extend(self.train_y_temp)
            self.test_y.extend(self.test_y_temp)

    def fit(self):
        pass

    def predict(self):
        self.test_pred_y = None
        pass

    def save(self):
        raise NotImplemented

    def load(self):
        raise NotImplemented

    def evaluate(self, threshold: float = 0.5, function='accuracy', average='binary'):
        if function == 'accuracy':
            print(accuracy_score(self.test_y, self.test_pred_y))
        elif function == 'confusion_matrix':
            print(confusion_matrix(self.test_y, self.test_pred_y))
        elif function == 'auc':
            print(roc_auc_score(self.test_y, self.test_pred_y))
        elif function == '':
            print(f1_score(self.test_y, self.test_pred_y))
        else:
            pass

class SVC_Model(Model):
    name = "SVC"
    # 模型参数
    param = {}
    description = "支持向量机分类"

    def __init__(self, x_all, y_all, kernel='rbf'):
        super(SVC_Model, self).__init__(x_all=x_all, y_all=y_all)
        self.kernel = kernel
        self.description = self.description + '-' + kernel

    def fit(self, **kwargs):
        k_single = kwargs.get('k_single', 0)
        k_pca = kwargs.get('k_pca', 1)
        fit_data = self.train_x.copy()

        # 数据归一化
        fit_data = minmax_scale(fit_data)

        # pca
        selection = SelectKBest(k=k_single)
        n_components = int(len(self.train_x[0]) * k_pca)
        # feature union
        # pca = PCA(n_components=n_components)
        # combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        # self.union = combined_features.fit(fit_data, self.train_y)
        self.pca = PCA(n_components=n_components).fit(fit_data)

        self.model = SVC(kernel=self.kernel, probability=True, gamma='auto')

        fit_data = self.pca.transform(fit_data)
        print('训练数据shape：', fit_data.shape)
        self.model.fit(fit_data, self.train_y)

    def predict(self, test_x=None):
        if test_x is not None:
            self.test_x = test_x
        test_data = self.test_x.copy()
        # 数据归一化
        test_data = minmax_scale(test_data)
        test_data = self.pca.transform(test_data)
        self.prob = self.model.predict_proba(test_data)
        self.pre_label = self.model.predict(test_data)
        self.cal_label = [np.argwhere(item == item.max())[0] for item in self.prob]
        '''
        for i in range(len(self.pre_label)):
            # print(self.pre_label[i])
            # print(self.cal_label[i])
            if int(self.pre_label[i]) != int(self.cal_label[i]):
                print(self.prob[i])
                print(str(self.pre_label[i]) + 'vs' + str(self.cal_label[i]))
        '''
        self.pre_label = np.array(self.cal_label)

    def evaluate(self, average='binary'):
        # print(self.test_y, self.prob)
        # 单独阈值效果指标
        print("accuracy: ", accuracy_score(self.test_y, self.pre_label))
        print("precision: ", precision_score(self.test_y, self.pre_label, average=average))
        print("recall: ", recall_score(self.test_y, self.pre_label, average=average))
        print("f1_score: ", f1_score(self.test_y, self.pre_label, average=average))
        print(confusion_matrix(self.test_y, self.pre_label))

        # 多阈值效果指标
        if average != 'binary':
            y_one_hot = label_binarize(self.test_y, np.arange(len(set(self.y_all))))
            self.test_y_multy = y_one_hot

        print("auc: ", roc_auc_score(self.test_y_multy, self.prob, average=average))


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"输出文件；默认标准输出设备")
    return parser


def main(options):

    meta = {}
    segment_path = []
    label = []
    audio = []
    with open('./meta.txt', 'r') as meta_data:
        for item in meta_data:
            item_group = item.strip().split('\t')
            if len(item_group) == 3:
                segment = item_group[0].split('/')[1]
                segment_path.append(segment)
                label.append(item_group[1])
                audio.append(item_group[2])

    meta['seg_path'] = segment_path
    meta['label'] = label
    meta['audio'] = audio
    meta = pd.DataFrame(meta)
    meta['label'] = preprocessing.LabelEncoder().fit_transform(meta['label'])

    return meta

def generate_featrue(meta=None):
    # check the meta data and the len(audio)
    print(meta.shape[0])
    audio_file = os.listdir('./audio')
    print(len(audio_file))
    print(len(set(meta['seg_path']).intersection(set(audio_file))))

    out_path = './feature_csv'
    error_list = []
    num_error = 0
    for item in audio_file:
        in_file = './audio/' + item
        out_file = './feature_csv/' + item + '.csv'
        try:
            os.system("/Users/mengshuai/Downloads/opensmile-2.3.0/SMILExtract -C /Users/mengshuai/Downloads/opensmile-2.3.0/config/IS09_emotion.conf -I %s -O %s" % (in_file, out_file))
        except:
            num_error += 1
            print('error', num_error)
            error_list.append(item)
            continue

        csv_reader = csv.reader(open(out_file))
        for row in csv_reader:
            pass
        # print(len(row))
        feature = row[1:-1]
        feature = np.array(feature)
        print(len(feature))

    print(error_list)

    pass

def get_all_feature(meta=None):
    feature_all = []
    file_name = []
    feature_file = os.listdir('./feature_csv')
    for item in feature_file:
        out_file = './feature_csv/' + item
        csv_reader = csv.reader(open(out_file))
        name = item.strip().split('.cs')[0]
        label = meta.loc[meta['seg_path'] == name, 'label']
        for row in csv_reader:
            pass
        # print(len(row))
        feature = row[1:-1]
        feature.extend(label)
        if len(feature) != 385:
            print('error')
            continue
        # if feature[-1] in [0, 1, 2, 3] or feature[-1] in ['0', '1', '2', '3']:
        feature_all.append(feature)
        file_name.append(name)
    # feature_all = np.array(feature_all)

    return feature_all, file_name

if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    meta = main(options)

    feature_all, file_name = get_all_feature(meta)
    # print(len(file_name), len(feature_all), len(feature_all[0]))

    x_all, y_all = [], []
    for item in feature_all:
        x_all.append(item[:-1])
        y_all.append(item[-1])

    svc_model = SVC_Model(x_all=x_all, y_all=y_all, kernel='rbf')
    print("训练集、测试集数量对比", len(svc_model.train_x), len(svc_model.test_x))
    svc_model.fit(k_single=0, k_pca=0.2)
    svc_model.predict()
    svc_model.evaluate(average='macro')