# coding=utf-8
import numpy as np
import codecs

TAGS = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
tags = ['B', 'I', 'E', 'S']


def get_corpus(filepath='../data/tiny.utf8'):
    """
    initialize only once
    :param filepath: corpus
    :return: char_list2D, tag_list2D
    """
    char_list2D = list()
    tag_list2D = list()
    with codecs.open(filepath, encoding='utf8') as fopen:
        char_list = list()
        tag_list = list()
        for line in fopen.readlines():
            if ' ' in line:
                char_list.append(line[0])
                tag_list.append(line[2])
            else:
                char_list2D.append(char_list)
                tag_list2D.append(tag_list)
                # new beginning
                char_list = list()
                tag_list = list()
        if len(char_list):
            char_list2D.append(char_list)
            tag_list2D.append(tag_list)
        return char_list2D, tag_list2D


def get_init_table(tag_list):
    return np.array([0.45, 0.05, 0.05, 0.45])


def get_transmission_table(char_list2D, tag_list2D):
    global char_dict
    # unique_chars = set(char_list)
    char_dict = dict()
    i = 0
    for char_list in char_list2D:
        for char in char_list:
            if not char_dict.__contains__(char):
                char_dict[char] = i
                i += 1
    T = len(char_dict)
    B = np.zeros([4, T])
    # count
    for char_list, tag_list in zip(char_list2D, tag_list2D):
        for char, tag in zip(char_list, tag_list):
            B[TAGS[tag], char_dict[char]] += 1
    # calculate probability
    for row in B:
        total = np.sum(row)
        if total != 0:
            row /= total
    return B


def get_transition_table(tag_list2D):
    A = np.zeros([4, 4])
    # count
    for tag_list in tag_list2D:
        for i in range(len(tag_list) - 1):
            tag = tag_list[i]
            next_tag = tag_list[i + 1]
            A[TAGS[tag]][TAGS[next_tag]] += 1
    # calculate probability
    for row in A:
        total = sum(row)
        if total != 0:
            row /= total
    return A


def getSeqFromStates(char_list):
    global char_dict
    result = np.zeros(len(char_list), dtype=int)
    i = 0
    for char in char_list:
        result[i] = char_dict[char]
        i += 1
    return result


def getSeqFromStates2D(char_list2D):
    result2D = []
    for char_list in char_list2D:
        result2D.append(getSeqFromStates(char_list))
    return result2D


def forward(A, B, pi, observationsSeq):
    """
    calculate alpha: matrix[T,4]
    :param A:
    :param B:
    :param pi:
    :param observationsSeq:
    :return:
    """
    T = len(observationsSeq)
    N = len(pi)
    alpha = np.zeros((T, N), dtype=float)
    alpha[0, :] = pi * B[:, observationsSeq[0]].T  # alpha[1,4] = pi[1,4] * B[1, 4]
    # 归一化因子
    c = np.zeros(T)
    c[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / c[0]  # 矩阵单元除
    # 递归传递
    for t in range(1, T):
        for n in range(0, N):
            alpha[t, n] = np.dot(alpha[t - 1, :], A[:, n]) * B[n, observationsSeq[t]]  # 使用内积简化代码
        c[t] = np.sum(alpha[t])
        if c[t] == 0: continue
        alpha[t] = alpha[t] / c[t]
    return alpha, c


def backward(A, B, pi, observationsSeq, c):
    """
    calculate beta: matrix[T,4]
    :param A: 
    :param B: 
    :param pi: 
    :param c: 归一化因子
    :param observationsSeq:
    :return: 
    """
    T = len(observationsSeq)
    N = len(pi)
    beta = np.zeros((T, N), dtype=float)
    beta[T - 1, :] = 1  # 最后概率累积一定为1
    for t in reversed(range(T - 1)):
        for n in range(N):
            beta[t, n] = np.sum(A[n, :] * B[:, observationsSeq[t + 1]] * beta[t + 1, :])
        if c[t + 1] == 0: continue
        beta[t] = beta[t] / c[t + 1]  # 矩阵单元除
    return beta


# def forwardAndBackword(observationsSeq, alpha, beta):
#     T = len(observationsSeq)
#     for t in range(T):
#         pom = np.sum(alpha[t, :] * beta[t, :])
#         print('pom = ', pom)


def get_epsilon(A, B, pi, alpha, beta, observationsSeq):
    """
    给定参数模型”入”,和观测序列O,在时刻t处在状态i且时刻为t+1处在状态为j的概率 ε(t)(i,j)= P(qt=Si,qt+1=Sj | O,λ)
    :param A:
    :param B:
    :param pi:
    :param alpha:
    :param beta:
    :param observationsSeq:
    :return:
    """
    T = len(observationsSeq)
    N = len(pi)
    xi = np.zeros((T - 1, N, N), dtype=float)
    for t in range(T - 1):
        denominator = np.sum(np.dot(alpha[t, :], A) * B[:, observationsSeq[t + 1]] * beta[t + 1, :])
        for i in range(N):  # 当前标签
            molecular = alpha[t, i] * A[i, :] * B[:, observationsSeq[t + 1]] * beta[t + 1, :]
            xi[t, i, :] = molecular / denominator  # todo: xi[t, i, :]
    return xi


def get_gamma(epsilon, alpha, beta, observationsSeq):
    """
    在给定模型参数和观测序列的前提下,t时刻处在状态i的概率. gamma(t)(i)
    根据epsilon就可以求出gamma，最后缺了一项要单独补上来
    :param epsilon:
    :param alpha:
    :param beta:
    :param observationsSeq:
    :return:
    """
    T = len(observationsSeq)
    gamma = np.sum(epsilon, axis=2)
    prod = (alpha[T - 1, :] * beta[T - 1, :])
    last_T_line = prod / np.sum(prod)
    if np.sum(prod) < 0.000000001:
        print('prod too small')
        print('prod = ', prod)
        print('alpha ', alpha)
        print('alpha[T - 1, :] = ', alpha[T - 1, :])
        print('beta[T - 1, :] = ', beta[T - 1, :])
    gamma = np.vstack((gamma, last_T_line))
    return gamma


def get_gamma2(alpha, beta):
    # 任意取一个时间,都可以获取分母,就是pom
    length = len(alpha)
    t = int(length / 2)   # todo
    pom = np.sum(alpha[t, :] * beta[t, :])
    if pom < 0.000000001:
        print('pom too small')
        print('pom = ', pom)
        print('alpha ', alpha)
        print('beta ', beta)
    gamma2 = alpha * beta / pom
    return gamma2


def train_with_EM_algo(A, B, pi, observationsSeq2D, criterion=0.001):
    while True:
        newA_numerator_list = []
        newB_numerator_list = []
        newA_denominator_list = []
        newB_denominator_list = []
        for observationsSeq in observationsSeq2D:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            alpha, c = forward(A, B, pi, observationsSeq)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            beta = backward(A, B, pi, observationsSeq, c)

            epsilon = get_epsilon(A, B, pi, alpha, beta, observationsSeq)

            gamma = get_gamma(epsilon, alpha, beta, observationsSeq)
            # gamma = get_gamma2(alpha, beta)

            newpi = gamma[0, :]
            for i in range(len(c)):
                if c[i] == 0:
                    c[i] = 1
            c = c.repeat(gamma.shape[1])
            c = c.reshape([-1,4])
            newA_numerator = np.sum(epsilon, axis=0)
            newA_denominator = np.sum(gamma[:-1, :]* c[:-1], axis=0).reshape(-1, 1)
            newA_denominator_list.append(newA_denominator)
            newA_numerator_list.append(newA_numerator)
            tmp_n = []
            tmp_d = []
            for k in range(B.shape[1]):  # T: number of unique chars
                mask = observationsSeq == k
                newB_numerator = np.sum(gamma[mask, :] * c[mask], axis=0)
                newB_denominator = np.sum(gamma * c, axis=0)
                tmp_d.append(newB_denominator)
                tmp_n.append(newB_numerator)
            newB_numerator_list.append(tmp_n)
            newB_denominator_list.append(tmp_d)


        newA = np.sum(newA_numerator_list, axis=0) / np.sum(newA_denominator_list, axis=0)
        newB_numerator_list = np.sum(newB_numerator_list, axis=0)   # newB_numerator_list: L*T*4 -> T*4
        newB_denominator_list = np.sum(newB_denominator_list, axis=0)
        newB = np.zeros(B.shape, dtype=float)
        for k in range(B.shape[1]):  # T: number of unique chars
            newB[:, k] = newB_numerator_list[k] / newB_denominator_list[k]
            # gamma: T*4; np.sum(gamma, axis=0): 1*4; B: 4*T; newB[:, k]: 4*1;

        if np.max(abs(pi - newpi)) < criterion and \
                np.max(abs(A - newA)) < criterion and \
                np.max(abs(B - newB)) < criterion:
            break

        A, B, pi = newA, newB, newpi
        for row in A:
            for col in range(len(A)):
                if np.isnan(row[col]):
                    row[col] = 0
        for row in B:
            for col in range(len(B)):
                if np.isnan(row[col]):
                    row[col] = 0
        for i in range(len(pi)):
            if np.isnan(pi[i]):
                pi[i] = 0
        print(A)
        print(B)
        print(pi)
    return A, B, pi


def train():
    char_list2D, tag_list2D = get_corpus()
    # transit states * states
    A = get_transition_table(tag_list2D)
    # emit states * observation
    B = get_transmission_table(char_list2D, tag_list2D)
    # init state probs
    pi = get_init_table(tag_list2D)
    observationsSeq = getSeqFromStates2D(char_list2D)
    print('A = ', A)
    print('B = ', B)
    print('pi = ', pi)
    # label在all_labels里面的下表构成了数组
    print('observationsSeq = ', observationsSeq)
    A, B, pi = train_with_EM_algo(A, B, pi, observationsSeq)
    print('A = ', A)
    print('B = ', B)
    print('pi = ', pi)


if __name__ == '__main__':
    train()
