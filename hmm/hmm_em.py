# coding=utf-8
import numpy as np
import codecs

TAGS = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
tags = ['B', 'I', 'E', 'S']


def get_corpus(filepath='../data/tiny.utf8'):
    """
    concatenate sentences into one
    :param filepath:
    :return:
    """
    char_list = list()
    tag_list = list()
    with codecs.open(filepath, encoding='utf8') as fopen:
        for line in fopen.readlines():
            if ' ' in line:
                char_list.append(line[0])
                tag_list.append(line[2])
    return char_list, tag_list


def get_init_table(tag_list):
    return np.array([tag_list[0] == 'B', 0, 0, tag_list[0] == 'S'])


def get_transmission_table(char_list, tag_list):
    global char_dict
    # unique_chars = set(char_list)
    char_dict = dict()
    i = 0
    for char in char_list:
        if not char_dict.__contains__(char):
            char_dict[char] = i
            i += 1
    T = len(char_list)
    B = np.zeros([4, T])
    # count
    for char, tag in zip(char_list, tag_list):
        B[TAGS[tag], char_dict[char]] += 1
    # calculate probability
    for row in B:
        total = np.sum(row)
        if total != 0:
            row /= total
    return B


def get_transition_table(tag_list):
    A = np.zeros([4, 4])
    # count
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


def forward(A, B, pi, observationsSeq):
    T = len(observationsSeq)
    N = len(pi)
    alpha = np.zeros((T, N), dtype=float)
    alpha[0, :] = pi * B[:, observationsSeq[0]]  # numpy可以简化循环
    for t in range(1, T):
        for n in range(0, N):
            alpha[t, n] = np.dot(alpha[t - 1, :], A[:, n]) * B[n, observationsSeq[t]]  # 使用内积简化代码
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
        print('pom = ', pom)


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


def train_with_EM_algo(A, B, pi, observationsSeq, criterion=0.001):
    while True:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        alpha = forward(A, B, pi, observationsSeq)

        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        beta = backward(A, B, pi, observationsSeq)

        # 根据公式求解epsilon(t)(i,j) = P(qt=Si,qt+1=Sj | O,λ)
        epsilon = get_epsilon(A, B, pi, alpha, beta, observationsSeq)

        # 根据xi就可以求出gamma，注意最后缺了一项要单独补上来
        # gamma = get_gamma(epsilon, alpha, beta, observationsSeq)
        gamma = get_gamma2(alpha, beta)

        newpi = gamma[0, :]
        # 当参数axis = 0时，求矩阵每一列上元素的和
        newA = np.sum(epsilon, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)

        newB = np.zeros(B.shape, dtype=float)
        for k in range(B.shape[1]):  # T: number of unique chars
            mask = observationsSeq == k
            newB[:, k] = np.sum(gamma[mask, :], axis=0) / np.sum(gamma, axis=0)
            # gamma: T*4; np.sum(gamma, axis=0): 1*4; B: 4*T; newB[:, k]: 4*1;

        if np.max(abs(pi - newpi)) < criterion and \
                np.max(abs(A - newA)) < criterion and \
                np.max(abs(B - newB)) < criterion:
            break

        A, B, pi = newA, newB, newpi
        print(A, B, pi)
    return A, B, pi


def train():
    char_list, tag_list = get_corpus()
    # transit states * states
    A = get_transition_table(tag_list)
    # emit states * observation
    B = get_transmission_table(char_list, tag_list)
    # init state probs
    pi = get_init_table(tag_list)
    observationsSeq = getSeqFromStates(char_list)
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
