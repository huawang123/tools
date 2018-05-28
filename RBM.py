# coding=utf-8
__author__ = "huawang"

import numpy as np


class RBM(object):
    def __init__(self, num_visible, num_hidden, learn_rate=0.1, learn_batch=1000):
        self.num_visible = num_visible  # 可视层神经元个数
        self.num_hidden = num_hidden  # 隐藏层神经元个数
        self.learn_rate = learn_rate  # 学习率
        self.learn_batch = learn_batch  # 每次根据多少样本进行学习

        '''初始化连接权重'''
        self.weights = 0.1 * \
                       np.random.randn(self.num_visible,
                                       self.num_hidden)  # 依据0.1倍的标准正太分布随机生成权重
        # 第一行插入全0，即偏置和隐藏层的权重初始化为0
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        # 第一列插入全0，即偏置和可视层的权重初始化为0
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def _logistic(self, x):
        '''直接使用1.0 / (1.0 + np.exp(-x))容易发警告“RuntimeWarning: overflowencountered in exp”，
           转换成如下等价形式后算法会更稳定
        '''
        return 0.5 * (1 + np.tanh(0.5 * x))

    def train(self, rating_data, max_steps=1000, eps=1.0e-4):
        '''迭代训练，得到连接权重
        '''
        for step in range(max_steps):  # 迭代训练多少次
            error = 0.0  # 误差平方和
            # 每次拿一批样本还调整权重
            for i in range(0, rating_data.shape[0], self.learn_batch):
                num_examples = min(self.learn_batch, rating_data.shape[0] - i)
                data = rating_data[i:i + num_examples, :]
                data = np.insert(data, 0, 1, axis=1)  # 第一列插入全1，即偏置的值初始化为1

                pos_hidden_activations = np.dot(data, self.weights)
                pos_hidden_probs = self._logistic(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
                # pos_associations=np.dot(data.T,pos_hidden_states)         #对隐藏层作二值化
                pos_associations = np.dot(data.T, pos_hidden_probs)  # 对隐藏层不作二值化

                neg_visible_activations = np.dot(
                    pos_hidden_states, self.weights.T)
                neg_visible_probs = self._logistic(neg_visible_activations)
                neg_visible_probs[:, 0] = 1  # 强行把偏置的值重置为1
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
                neg_hidden_probs = self._logistic(neg_hidden_activations)
                # neg_hidden_states=neg_hidden_probs>np.random.rand(num_examples,self.num_hidden+1)
                # neg_associations=np.dot(neg_visible_probs.T,neg_hidden_states)      #对隐藏层作二值化
                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)  # 对隐藏层不作二值化

                # 更新权重。另外一种尝试是带冲量的梯度下降，即本次前进的方向是本次梯度与上一次梯度的线性
                # 加权和（这样的话需要额外保存上一次的梯度）
                self.weights += self.learn_rate * \
                                (pos_associations - neg_associations) / num_examples

                # 计算误差平方和
                error += np.sum((data - neg_visible_probs) ** 2)
            if error < eps:  # 所有样本的误差平方和低于阈值于终止迭代
                break
            print('iteration %d, error is %f' % (step, error))

    def getHidden(self, visible_data):
        '''根据输入层得到隐藏层
           visible_data是一个matrix，每行代表一个样本
        '''
        num_examples = visible_data.shape[0]
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)  # 第一列插入偏置
        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self._logistic(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(
            num_examples, self.num_hidden + 1)
        hidden_states = hidden_states[:, 1:]  # 即首列删掉，即把偏置去掉
        return hidden_states

    def getVisible(self, hidden_data):
        '''根据隐藏层得到输入层
           hidden_data是一个matrix，每行代表一个样本
        '''
        num_examples = hidden_data.shape[0]
        visible_states = np.ones((num_examples, self.num_visible + 1))
        hidden_data = np.insert(hidden_data, 0, 1, axis=1)
        visible_activations = np.dot(hidden_data, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(
            num_examples, self.num_visible + 1)
        visible_states = visible_states[:, 1:]
        return visible_states

    def predict(self, visible_data):
        num_examples = visible_data.shape[0]
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)  # 第一列插入偏置
        '''forward'''
        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self._logistic(hidden_activations)
        # hidden_states[:, :] = hidden_probs > np.random.rand(
        #     num_examples, self.num_hidden + 1)
        '''backward'''
        visible_states = np.ones((num_examples, self.num_visible + 1))
        # visible_activations = np.dot(hidden_states, self.weights.T)  #对隐藏层作二值化
        visible_activations = np.dot(hidden_probs, self.weights.T)  # 对隐藏层不作二值化
        visible_probs = self._logistic(visible_activations)  # 直接返回可视层的概率值

        return visible_probs[:, 1:]  # 把第0列(偏置)去掉


if __name__ == '__main__':
    rbm = RBM(num_visible=6, num_hidden=2, learn_rate=0.08, learn_batch=1000)
    rating_data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [
        0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]])
    rbm.train(rating_data, max_steps=5000, eps=1.0e-4)
    print('weight:\n', rbm.weights)
    rating = np.array([[0, 0, 0, 0.9, 0.7, 0]])  # 评分需要做归一化。该用户喜欢第四、五项
    hidden_data = rbm.getHidden(rating)
    print('hidden_data:\n', hidden_data)
    visible_data = rbm.getVisible(hidden_data)
    print('visible_data:\n', visible_data)
    predict_data = rbm.predict(rating)
    print('推荐得分:')
    for i, score in enumerate(predict_data[0, :]):
        print(i, score )
