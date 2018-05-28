
# coding=utf-8
__author__ = 'huawang'

import time
import operator
def PersonalRank(G, alpha, root, max_step):
    rank = dict()
    rank = {x: 0 for x in G.keys()}
    rank[root] = 1
    # 开始迭代
    begin = time.time()
    for k in range(max_step):
        tmp = {x: 0 for x in G.keys()}
        # 取节点i和它的出边尾节点集合ri
        for i, ri in G.items():
            # 取节点i的出边的尾节点j以及边E(i,j)的权重wij, 边的权重都为1，归一化之后就上1/len(ri)
            for j, wij in ri.items():
                # i是j的其中一条入边的首节点，因此需要遍历图找到j的入边的首节点，
                # 这个遍历过程就是此处的2层for循环，一次遍历就是一次游走
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
        # 我们每次游走都是从root节点出发，因此root节点的权重需要加上(1 - alpha)
        tmp[root] += (1 - alpha)
        rank = tmp
    end = time.time()
    print('use time', end - begin)

    li = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for ele in li:
        print("%s:%.3f, \t" % (ele[0], ele[1]))


    return rank


if __name__ == '__main__':
    alpha = 0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}

    PersonalRank(G, alpha, 'b', 50)  # 从'b'节点开始游走

    
    
# ######################################
# coding=utf-8
__author__ = 'orisun'
import operator
import numpy as np
from numpy.linalg import solve
import time
from scipy.sparse.linalg import gmres, lgmres
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    alpha = 0.8
    vertex = ['A', 'B', 'C', 'a', 'b', 'c', 'd']
    M = np.matrix([[0, 0, 0, 0.5, 0, 0.5, 0],
                   [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
                   [0, 0, 0, 0, 0, 0.5, 0.5],
                   [0.5, 0.5, 0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0, 0, 0],
                   [0.333, 0.333, 0.333, 0, 0, 0, 0],
                   [0, 0.5, 0.5, 0, 0, 0, 0]])
    # print np.eye(n) - alpha * M.T
    r0 = np.matrix([[0], [0], [0], [0], [1], [0], [0]])  # 从'b'节点开始游走
    n = M.shape[0]

    # 直接解线性方程法
    A = np.eye(n) - alpha * M.T
    b = (1 - alpha) * r0
    begin = time.time()
    r = solve(A, b)
    end = time.time()
    print('use time', end - begin)
    rank = {}
    for j in range(n):
        rank[vertex[j]] = r[j]
    li = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for ele in li:
        print("%s:%.3f, \t" % (ele[0], ele[1]))

    # 采用CSR法对稀疏矩阵进行压缩存储，然后解线性方程
    A = np.eye(n) - alpha * M.T
    b = (1 - alpha) * r0
    data = list()
    row_ind = list()
    col_ind = list()
    for row in range(n):
        for col in range(n):
            if (A[row, col] != 0):
                data.append(A[row, col])
                row_ind.append(row)
                col_ind.append(col)
    AA = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    begin = time.time()
    # 系数矩阵很稀疏时采用gmres方法求解。解方程的速度比上面快了许多
    r = gmres(AA, b, tol=1e-08, maxiter=1)[0]
    # r = lgmres(AA, (1 - alpha) * r0, tol=1e-08,maxiter=1)[0]  #lgmres用来克服gmres有时候不收敛的问题，会在更少的迭代次数内收敛
    end = time.time()
    print('use time', end - begin)
    rank = {}
    for j in range(n):
        rank[vertex[j]] = r[j]
    li = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for ele in li:
        print("%s:%.3f, \t" % (ele[0], ele[1]))

    # 求逆矩阵法。跟gmres解方程的速度相当
    A = np.eye(n) - alpha * M.T
    b = (1 - alpha) * r0
    begin = time.time()
    r = A.I * b
    end = time.time()
    print('use time', end - begin)
    rank = {}
    for j in range(n):
        rank[vertex[j]] = r[j, 0]
    li = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for ele in li:
        print("%s:%.3f, \t" % (ele[0], ele[1]))

    # 实际上可以一次性计算出从任意节点开始游走的PersonalRank结果。从总体上看，这种方法是最快的
    A = np.eye(n) - alpha * M.T
    begin = time.time()
    D = A.I
    end = time.time()
    print('use time', end - begin)
    for j in range(n):
        print(vertex[j] + "\t")
        score = {}
        total = 0.0  # 用于归一化
        for i in range(n):
            score[vertex[i]] = D[i, j]
            total += D[i, j]
        li = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
        for ele in li:
            print("%s:%.3f, \t" % (ele[0], ele[1] / total))
