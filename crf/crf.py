import codecs
from pprint import pprint
import numpy as np
import json
import matplotlib.pyplot as plt

tags = ['B', 'I', 'E', 'S']
bias = 2


def get_macros(template='../data/template.utf8'):
    """
    initialize only once
    :param template: filepath
    :return:
    """
    macros = dict()
    with open(template) as fopen:
        for line in fopen:
            if ':' not in line:
                continue
            strings = line.split(':%x')
            id = strings[0]  # U00 or B01
            eles = strings[1].split('/%x')
            relative_pos = [eval(e)[0] for e in eles]  # I am sure there is no col info
            macros[id] = relative_pos
    return macros


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
                char_list.insert(0, '_I-1')
                char_list.insert(0, '_I-2')
                char_list.append('_I+1')
                char_list.append('_I+2')
                tag_list.insert(0, '_T-1')
                tag_list.insert(0, '_T-2')
                tag_list.append('_T+1')
                tag_list.append('_T+2')
                char_list2D.append(char_list)
                tag_list2D.append(tag_list)
                # new beginning
                char_list = list()
                tag_list = list()
        if len(char_list):
            char_list.insert(0, '_I-1')
            char_list.insert(0, '_I-2')
            char_list.append('_I+1')
            char_list.append('_I+2')
            tag_list.insert(0, '_T-1')
            tag_list.insert(0, '_T-2')
            tag_list.append('_T+1')
            tag_list.append('_T+2')
            char_list2D.append(char_list)
            tag_list2D.append(tag_list)
        return char_list2D, tag_list2D


def generate_feature_functions(char_list2D, tag_list2D):
    """
    only for initialization
    :param char_list2D:
    :return: feature_funtions<dict>, total_num of feature_funtions
    """
    global bias, macros
    total_num = 0  # num_params == num_feature_functions
    macros = dict(macros)
    feature_functions = dict()
    for id_macro, relative_pos in macros.items():
        for char_list, tag_list in zip(char_list2D, tag_list2D):
            for i in range(len(char_list) - bias * 2):
                observation = ''
                if id_macro[0] == 'U':  # remove observation in B macros -> transition table
                    for pos in relative_pos:
                        observation += char_list[i + bias + pos]
                tag = tag_list[i + bias]
                pre_tag = tag_list[i + bias - 1] if id_macro[0] == 'B' else ''
                feature_functions[id_macro + observation + pre_tag + tag] = 0
                total_num += 1
    return feature_functions, total_num


def viterbi_process(feature_functions, char_list):
    """
    decoding a sentence
    :param feature_functions: [whole won't affect efficiency] modified in each iteration of training
    :param char_list:
    :return: predicted_tag_list
    """
    global tags, bias, macros
    feature_functions = dict(feature_functions)
    char_list = list(char_list)
    macros = dict(macros)
    score_matrix = np.zeros([len(tags), len(char_list) - bias * 2])
    track_matrix = np.zeros([len(tags), len(char_list) - bias * 2])
    for i in range(len(tags)):  # the first col is NIL
        track_matrix[i, 0] = -1
    # forward
    # the first char
    first_char_index = bias  # 0+bias
    for row_score in range(len(tags)):
        tag = tags[row_score]
        score = 0
        for id_macro, relative_pos in macros.items():
            o = ''
            for pos in relative_pos:
                observe_index = first_char_index + pos
                o += char_list[observe_index]
            my_pre_tag = '' if id_macro[0] == 'U' else '_T-1'
            try:
                param = feature_functions[id_macro + o + my_pre_tag + tag]
            except Exception:
                param = 0
            score += param
        score_matrix[row_score, 0] = score
    # for the rest chars
    for this_char_index in range(first_char_index + 1, len(char_list) - bias):  # col
        for row_score in range(len(tags)):
            tag = tags[row_score]
            max_score = -1e10
            index_track = 0
            for row_track in range(len(tags)):
                pre_tag = tags[row_track]
                score = score_matrix[row_track, this_char_index - 1 - bias]  # sum of previous chars
                for id_macro, relative_pos in macros.items():
                    o = ''
                    for pos in relative_pos:
                        observe_index = this_char_index + pos
                        o += char_list[observe_index]
                    my_pre_tag = '' if id_macro[0] == 'U' else pre_tag
                    try:
                        param = feature_functions[id_macro + o + my_pre_tag + tag]
                    except Exception:
                        param = 0
                    score += param
                if score > max_score:
                    max_score = score
                    index_track = row_track
            track_matrix[row_score, this_char_index - bias] = index_track
            score_matrix[row_score, this_char_index - bias] = max_score
    # backward
    l = len(char_list) - bias * 2
    l -= 1
    last_column = score_matrix[:, l]
    pre_tag_index = last_column.argmax(0)
    predicted_tag_list = tags[pre_tag_index]
    while l > 0:
        pre_tag_index = int(track_matrix[pre_tag_index, l])
        l -= 1
        predicted_tag_list = tags[pre_tag_index] + predicted_tag_list
    predicted_tag_list = list(predicted_tag_list)  # string to char[]
    predicted_tag_list.insert(0, '_T-1')
    predicted_tag_list.insert(0, '_T-2')
    predicted_tag_list.append('_T+1')
    predicted_tag_list.append('_T+2')
    return predicted_tag_list


def viterbi_process2D(feature_functions, char_list2D):
    """
    2D version of viterbi_process
    :param feature_functions: the whole
    :param char_list2D:
    :return: predicted_tag_list2D
    """
    predicted_tag_list2D = list()
    for char_list in char_list2D:
        predicted_tag_list = viterbi_process(feature_functions, char_list)
        predicted_tag_list2D.append(predicted_tag_list)
    return predicted_tag_list2D


def train_param(feature_functions, right_counts, char_list2D, predicted_tag_list2D):
    """
    modify feature_functions
    :param feature_functions: whole
    :param right_counts: whole
    :param char_list2D:
    :param predicted_tag_list2D:
    :return: feature_functions: modified, whole
    """
    wrong_counts = get_counts(char_list2D, predicted_tag_list2D)
    # parse all feature_functions
    for key in feature_functions.keys():
        wrong_count = wrong_counts[key] if wrong_counts.__contains__(key) else 0
        right_count = right_counts[key] if right_counts.__contains__(key) else 0
        feature_functions[key] += right_count - wrong_count
    return feature_functions


def get_counts(char_list2D, tag_list2D):
    """
    structure of right_counts == structure of init_feature_functions
    :param char_list2D:
    :param tag_list2D:
    :return:
    """
    global bias, macros
    macros = dict(macros)
    feature_functions = dict()
    for id_macro, relative_pos in macros.items():
        for char_list, tag_list in zip(char_list2D, tag_list2D):
            for i in range(len(char_list) - bias * 2):
                observation = ''
                if id_macro[0] == 'U':  # remove observation in B macros -> transition table
                    for pos in relative_pos:
                        observation += char_list[i + bias + pos]
                pre_tag = '' if id_macro[0] == 'U' else tag_list[i + bias - 1]
                tag = tag_list[i + bias]
                key = id_macro + observation + pre_tag + tag
                if feature_functions.__contains__(key):
                    feature_functions[id_macro + observation + pre_tag + tag] += 1
                else:
                    feature_functions[id_macro + observation + pre_tag + tag] = 1
    return feature_functions


def filter_features(feature_functions):
    right_counts = dict()
    new_feature_functions = dict()
    total_num = 0
    for key in feature_functions.keys():
        if key[0] == 'B' or feature_functions[key] > 2:  # todo: threshold
            right_counts[key] = feature_functions[key]
            new_feature_functions[key] = 0
            total_num += 1
    return new_feature_functions, right_counts, total_num


def calculate_accuracy(tag_list, predicted_tag_list):
    global extended_tags
    extended_tags = ['_T-2', '_T-1', '_T+1', '_T+2']
    correct_counts = 0
    total = len(tag_list) - bias * 2
    for x, y in zip(tag_list, predicted_tag_list):
        if x == y and not extended_tags.__contains__(x):
            correct_counts += 1
    return correct_counts, total


def calculate_accuracy2D(tag_list2D, predicted_tag_list2D):
    c, t = (0, 0)
    for tag_list, predicted_tag_list in zip(tag_list2D, predicted_tag_list2D):
        correct_counts, total = calculate_accuracy(tag_list, predicted_tag_list)
        c += correct_counts
        t += total
    return c / t


def store_model(feature_functions, filepath='../data/crf_model.json'):
    json.dump(feature_functions, codecs.open(filepath, 'w', encoding='utf8'))


def import_model(filepath='../data/crf_model.json'):
    return json.load(codecs.open(filepath, 'r', encoding='utf8'))


def get_counts1D(char_list, tag_list):
    global bias, macros
    macros = dict(macros)
    feature_functions = dict()
    for id_macro, relative_pos in macros.items():
        for i in range(len(char_list) - bias * 2):
            observation = ''
            if id_macro[0] == 'U':  # remove observation in B macros -> transition table
                for pos in relative_pos:
                    observation += char_list[i + bias + pos]
            pre_tag = '' if id_macro[0] == 'U' else tag_list[i + bias - 1]
            tag = tag_list[i + bias]
            key = id_macro + observation + pre_tag + tag
            if feature_functions.__contains__(key):
                feature_functions[id_macro + observation + pre_tag + tag] += 1
            else:
                feature_functions[id_macro + observation + pre_tag + tag] = 1
    return feature_functions


def train_param_each_sentence(training_set='../data/train.utf8', model='../data/crf_model.json'):
    global macros
    # Fixed:
    macros = get_macros(template='../data/template2.utf8')
    print('generated macros')
    char_list2D, tag_list2D = get_corpus(training_set)
    # the previous 2 are unrelated
    print('there are %d sentences' % len(char_list2D))
    right_counts = get_counts(char_list2D, tag_list2D)
    feature_functions, right_counts, total_num = filter_features(right_counts)
    right = []
    for j in range(len(char_list2D)):
        char_list = char_list2D[j]
        tag_list = tag_list2D[j]
        right_count = get_counts1D(char_list, tag_list)
        right.append(right_count)
    print('finish configuration. start training')
    # To train:
    # feature_functions, total_num = generate_feature_functions(char_list2D, tag_list2D)  # initial params as 0
    print('there are %d feature functions in this model' % total_num)
    accuracy = []
    for i in range(200):
        for j in range(len(char_list2D)):
            char_list = char_list2D[j]
            tag_list = tag_list2D[j]
            predicted_tag_list = viterbi_process(feature_functions, char_list)
            # print(calculate_accuracy2D([tag_list], [predicted_tag_list]))
            feature_functions = train_param(feature_functions, right[j], [char_list], [predicted_tag_list])
        predicted_tag_list2D = viterbi_process2D(feature_functions, char_list2D)
        a = calculate_accuracy2D(tag_list2D, predicted_tag_list2D)
        accuracy.append(a)
        print(a)
        if i % 10:
            store_model(feature_functions, model)
    store_model(feature_functions, model)
    return accuracy


def initial_train(training_set='../data/train.utf8', model='../data/crf_model.json'):
    global macros
    # Fixed:
    macros = get_macros(template='../data/template2.utf8')
    print('generated macros')
    char_list2D, tag_list2D = get_corpus(training_set)
    # the previous 2 are unrelated
    print('there are %d sentences' % len(char_list2D))
    right_counts = get_counts(char_list2D, tag_list2D)
    feature_functions, right_counts, total_num = filter_features(right_counts)
    print('finish configuration. start training')
    # To train:
    # feature_functions, total_num = generate_feature_functions(char_list2D, tag_list2D)  # initial params as 0
    print('there are %d feature functions in this model' % total_num)
    previous_predicted_tag_list2D = None
    repetitive_num = 0
    accuracy = []
    for i in range(200):
        predicted_tag_list2D = viterbi_process2D(feature_functions, char_list2D)
        # print(predicted_tag_list2D)
        a = calculate_accuracy2D(tag_list2D, predicted_tag_list2D)
        accuracy.append(a)
        print(a)
        if predicted_tag_list2D == tag_list2D:
            break
        if predicted_tag_list2D == previous_predicted_tag_list2D:
            repetitive_num += 1
        else:
            repetitive_num = 0
            previous_predicted_tag_list2D = predicted_tag_list2D
        if repetitive_num == 5:
            break
        feature_functions = train_param(feature_functions, right_counts, char_list2D, predicted_tag_list2D)
        if i % 10:
            store_model(feature_functions, model)
    store_model(feature_functions, model)
    return accuracy


def plot_result(accuracy):
    plt.figure()
    plt.plot(accuracy)
    plt.show()
    plt.savefig('../data/result.png')



def continue_train(model='../data/crf_model.json'):
    global macros
    # Fixed:
    macros = get_macros(template='../data/template2.utf8')
    print('generated macros')
    char_list2D, tag_list2D = get_corpus('../data/train.utf8')
    # the previous 2 are unrelated
    print('there are %d sentences' % len(char_list2D))
    right_counts = get_counts(char_list2D, tag_list2D)
    feature_functions = import_model(model)
    print('finish configuration. start training')
    # To train:
    # feature_functions, total_num = generate_feature_functions(char_list2D, tag_list2D)
    # print('there are %d feature functions in this model' % total_num)
    previous_predicted_tag_list2D = None
    repetitive_num = 0
    for i in range(100):
        predicted_tag_list2D = viterbi_process2D(feature_functions, char_list2D)
        # print(predicted_tag_list2D)
        print(calculate_accuracy2D(tag_list2D, predicted_tag_list2D))
        if predicted_tag_list2D == tag_list2D:
            break
        if predicted_tag_list2D == tag_list2D:
            break
        if predicted_tag_list2D == previous_predicted_tag_list2D:
            repetitive_num += 1
        else:
            repetitive_num = 0
            previous_predicted_tag_list2D = predicted_tag_list2D
        if repetitive_num == 5:
            break
        feature_functions = train_param(feature_functions, right_counts, char_list2D, predicted_tag_list2D)
        if i % 10:
            store_model(feature_functions, model)
    store_model(feature_functions, model)


if __name__ == '__main__':
    accuracy = train_param_each_sentence()
    plot_result(accuracy)
    # accuracy = []
    # with open('../data/test.json') as fopen:
    #     for line in fopen:
    #         accuracy.append(float(line))
    # plt.figure()
    # plt.plot(accuracy)
    # plt.show()