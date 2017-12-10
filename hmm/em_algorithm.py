# coding=utf-8
import numpy as np
import sys

import ai_lab_2.tool.file_tool as file_tool
import ai_lab_2.tool.global_variable as global_variable

reload(sys)
sys.setdefaultencoding("utf-8")


def getSeqFromStates(all_labels, observations_labels):
    state_index_map = {}
    i = 0
    for state in all_labels:
        state_index_map[state] = i
        i += 1
    result = np.zeros((len(observations_labels)), dtype=float)
    index = 0
    for observationsState in observations_labels:
        result[index] = state_index_map[observationsState]
        index += 1
    return result


def forward(A, B, pi, observationsSeq):
    T = len(observationsSeq)
    N = len(pi)
    alpha = np.zeros((T, N), dtype=float)
    alpha[0, :] = pi * B[:, observationsSeq[0]]  # numpy可以简化循环
    for t in range(1, T):
        for n in range(0, N):
            alpha[t, n] = np.dot(alpha[t - 1, :], A[:, n]) * B[n, observationsSeq[t]]  # 使用内积简化代码
    pom = np.sum(alpha[T - 1, :])
    return alpha


# 计算公式中的beita二维数组
def backward(A, B, pi, observationsSeq):
    T = len(observationsSeq)
    N = len(pi)
    beta = np.zeros((T, N), dtype=float)
    beta[T - 1, :] = 1
    for t in reversed(range(T - 1)):
        for n in range(N):
            beta[t, n] = np.sum(A[n, :] * B[:, observationsSeq[t + 1]] * beta[t + 1, :])
    pom = np.sum(pi * B[:, 0] * beta[0, :])
    return beta


def forwardAndBackword(observationsSeq, alpha, beta):
    T = len(observationsSeq)
    for t in range(T):
        pom = np.sum(alpha[t, :] * beta[t, :])
        print "pom = ", pom


# 给定参数模型”入”,和观测序列O,在时刻t处在状态i且时刻为t+1处在状态为j的概率 ε(t)(i,j)
def get_epsilon(A, B, pi, alpha, beta, observationsSeq):
    T = len(observationsSeq)
    N = len(pi)
    # 根据公式求解XIt(i,j) = P(qt=Si,qt+1=Sj | O,λ)
    xi = np.zeros((T - 1, N, N), dtype=float)
    # t = 0,1,2,...,T-2 因为T-1后面没有状态了
    for t in range(T - 1):
        # 计算一个t对应的矩阵
        # fen mu down
        denominator = np.sum(np.dot(alpha[t, :], A) * B[:, observationsSeq[t + 1]] * beta[t + 1, :])
        for i in range(N):
            # fen zi up
            molecular = alpha[t, i] * A[i, :] * B[:, observationsSeq[t + 1]] * beta[t + 1, :]
            xi[t, i, :] = molecular / denominator
    return xi


# 在给定模型参数和观测序列的前提下,t时刻处在状态i的概率. gamma(t)(i)
def get_gamma(epsilon, alpha, beta, observationsSeq):
    T = len(observationsSeq)
    # 根据xi就可以求出gamma，注意最后缺了一项要单独补上来
    # axis = 2时，对矩阵每个元素进行求和，即对你矩阵中每个列表内的元素求和
    gamma = np.sum(epsilon, axis=2)
    prod = (alpha[T - 1, :] * beta[T - 1, :])
    last_T_line = prod / np.sum(prod)
    if np.sum(prod) < 0.000000001:
        print "prod too small"
        print "prod = ", prod
        print "alpha ", alpha
        print "alpha[T - 1, :] = ", alpha[T - 1, :]
        print "beta[T - 1, :] = ", beta[T - 1, :]
    gamma = np.vstack((gamma, last_T_line))
    return gamma


def get_gamma2(alpha, beta):
    # 任意取一个时间,都可以获取分母,就是pom
    length = len(alpha)
    t = length / 2
    pom = np.sum(alpha[t, :] * beta[t, :])
    if pom < 0.000000001:
        print "pom too small"
        print "pom = ", pom
        print "alpha ", alpha
        print "beta ", beta
    gamma2 = alpha * beta / pom
    return gamma2


def train_with_EM_algo(A, B, pi, observationsSeq, criterion=0.001):
    T = len(observationsSeq)
    N = len(pi)

    while True:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        alpha = forward(A, B, pi, observationsSeq)

        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        beta = backward(A, B, pi, observationsSeq)

        # 根据公式求解epsilon(t)(i,j) = P(qt=Si,qt+1=Sj | O,λ)
        epsilon = get_epsilon(A, B, pi, alpha, beta, observationsSeq)

        # 根据xi就可以求出gamma，注意最后缺了一项要单独补上来
        gamma = get_gamma(epsilon, alpha, beta, observationsSeq)
        gamma = get_gamma2(alpha, beta)

        newpi = gamma[0, :]
        # 当参数axis = 0时，求矩阵每一列上元素的和
        newA = np.sum(epsilon, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)

        newB = np.zeros(B.shape, dtype=float)
        for k in range(B.shape[1]):
            mask = observationsSeq == k
            newB[:, k] = np.sum(gamma[mask, :], axis=0) / np.sum(gamma, axis=0)

        if np.max(abs(pi - newpi)) < criterion and \
                        np.max(abs(A - newA)) < criterion and \
                        np.max(abs(B - newB)) < criterion:
            break

        A, B, pi = newA, newB, newpi
    return A, B, pi


def test_forward():
    # transit states * states
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]])
    # emit states * label
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]])
    # init state probs
    pi = np.array([0.2, 0.4, 0.4])
    observationsLabels = ['red', 'white', 'red', 'white', 'red', 'red']
    all_labels = ['red', 'white']
    observationsSeq = getSeqFromStates(all_labels, observationsLabels)

    print "A = ", A
    print "B = ", B
    print "pi = ", pi
    # label在all_labels里面的下表构成了数组
    print "observationsSeq = ", observationsSeq
    alpha = forward(A, B, pi, observationsSeq)
    print "alpha = ", alpha
    beta = backward(A, B, pi, observationsSeq)
    print "beta = ", beta
    forwardAndBackword(observationsSeq, alpha, beta)
    epsilon = get_epsilon(A, B, pi, alpha, beta, observationsSeq)
    print "epsilon = ", epsilon
    gamma = get_gamma(epsilon, alpha, beta, observationsSeq)
    print "gamma = ", gamma
    gamma2 = get_gamma2(alpha, beta)
    print "gamma2 = ", gamma2
    print "diff gamma-gamma2", gamma - gamma2

if __name__ == '__main__':
    test_forward()
    test_file()
