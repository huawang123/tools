# coding=utf-8
__author__ = "huawang"

import numpy as np
import sys

class OneDimClassifier(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.split = 0
        # direct = True时:小于分割线的为正样本，大于分割线的为负样本
        self.direct = True

    def train(self, w):
        '''根据各样本的分类及权重，选择使总误差最小的分割线
        '''
        # 将x,y和w进行绑定
        sx = np.concatenate((self.x, self.y.reshape(
            (self.y.shape[0], 1)), w.reshape((w.shape[0], 1))), axis=1)
        # 样本排序
        sx = [sx[i] for i in np.argsort(sx, axis=0)[:, 0]]

        prevY = sx[0][1]
        minErr = sys.float_info.max
        for i in range(1, len(sx)):
            # 尝试在每一个y值发生变化的点上进行分割
            if sx[i][1] != prevY:
                err = 0.0
                d = True
                for j in range(i):
                    if sx[j][1] != 1:
                        err += sx[j][2]
                for j in range(i, len(sx)):
                    if sx[j][1] != -1:
                        err += sx[j][2]
                if err > 0.5:
                    err = 1.0 - err
                    d = False
                if err < minErr:
                    self.split = (sx[i][0] + sx[i - 1][0]) / 2
                    minErr = err
                    self.direct = d
            prevY = sx[i][1]

        print ('split=', self.split)
        return minErr

    def predict(self, x):
        pre_y = np.zeros(x.shape[0])
        factor = 1 if self.direct else -1
        for i in range(x.shape[0]):
            if x[i][0] <= self.split:
                pre_y[i] = 1 * factor
            else:
                pre_y[i] = -1 * factor
        return pre_y


class Adboost(object):

    def __init__(self, x, y, WeakClassifier, M):
        # 输入样本的特征向量
        self.x = np.array(x)
        # 输入样本的分类标识，用1或-1表示
        self.y = np.array(y)
        # 初始化每个样本的权重
        self.w = np.array([1.0 / self.x.shape[0]
                           for i in range(self.x.shape[0])])
        # 弱分类器的构建器
        self.WeakClassifier = WeakClassifier
        # 弱分类器的数目上限
        self.M = M
        # 实际使用的弱分类器的数目
        self.Q = 0
        # 弱分类器集合
        self.G = []
        # 各个弱分类器的权重
        self.alpha = []

    def predict(self, x):
        '''预测分类
        '''
        if self.Q <= 0:
            raise Exception("have not train before predict")
        pre_y = np.zeros(x.shape[0])
        for i in range(self.Q):
            pre_y += self.G[i].predict(x) * self.alpha[i]
        return np.sign(pre_y)

    def train(self):
        '''训练各个弱分类器及其权重
        '''
        for i in range(self.M):
            # 用WeakClassifier初始化第一个弱分类器
            self.G.append(self.WeakClassifier(self.x, self.y))
            # 用当前各个样本的权重训练当前的弱分类器，并返回错误率
            e = self.G[i].train(self.w)
            # e不能等于0.5
            while e == 0.5:
                e += np.random.uniform(-0.1, 0.1)
            # 计算当前分类器的权重
            a = 1.0 / 2.0 * np.log((1 - e) / e)
            self.alpha.append(a)
            # 用当前的分类器预测每一个样本的分类
            pre_y = self.G[i].predict(self.x)
            # 计算下一轮中各个样本的权重
            self.w *= np.exp(-a * self.y * pre_y)
            # 对权重进行归一化，使其是一个概率分布
            self.w /= self.w.sum()
            self.Q = i + 1
            errnum = (self.y != self.predict(self.x)).sum()
            if errnum == 0:
                print (self.Q, "week classifiers is enough to make the error of train set to zero")
                break
        # 返回在训练集上的错误率
        return 1.0 * (self.y != self.predict(self.x)).sum() / self.x.shape[0]

if __name__ == '__main__':
    x = [[0], [2], [4], [6], [8], [1],  [3],  [5],  [7],  [9]]
    y = [1,    1,   -1,  1,   1,   1,    -1,   -1,   1,    -1]
    boost = Adboost(x, y, OneDimClassifier, 5)
    errratio = boost.train()
    print ('week classifier weight:', boost.alpha)
    print ('error ratio:', errratio)
    print ('tag of 4.3 is', boost.predict(np.array([[4.3]])))
