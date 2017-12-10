import codecs
import numpy as np
from pprint import pprint
import json

zero = -3.14e+100
TAGS = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
tags = ['B', 'I', 'E', 'S']


def get_corpus(filepath='../data/train.utf8'):
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


def get_init_table(tag_list2D):
    total = len(tag_list2D)
    count_B = 0
    count_S = 0
    for tag_list in tag_list2D:
        if tag_list[0] == 'B':
            count_B += 1
        else:
            count_S += 1
    # return [count_B / total, count_S / total]
    return [np.log(count_B / total), zero, zero, np.log(count_S / total)]


def get_transmission_table(char_list2D, tag_list2D):
    transmission_table = [dict(), dict(), dict(), dict()]
    for char_list, tag_list in zip(char_list2D, tag_list2D):
        for char, tag in zip(char_list, tag_list):
            if transmission_table[TAGS[tag]].__contains__(char):
                transmission_table[TAGS[tag]][char] += 1
            else:
                transmission_table[TAGS[tag]][char] = 1
    for obs in transmission_table:
        total = 0
        for k, c in obs.items():
            total += c
        for k, c in obs.items():
            obs[k] = np.log(c / total)
    return transmission_table


def get_transition_table(tag_list2D):
    transition_table = np.zeros([4, 4])
    for tag_list in tag_list2D:
        for i in range(len(tag_list) - 1):
            tag = tag_list[i]
            next_tag = tag_list[i + 1]
            transition_table[TAGS[tag]][TAGS[next_tag]] += 1
    for row in transition_table:
        total = sum(row)
        for i in range(len(row)):
            row[i] = np.log(row[i] / total) if row[i] != 0 else zero
    return transition_table


def viterbi_decoding(char_list):
    length = len(char_list)
    score_matrix = np.zeros([4, length])
    track_matrix = np.zeros([4, length])
    for row in track_matrix:
        row[0] = -1
    # forward
    # the first char
    for row in range(4):
        try:
            score_matrix[row, 0] = init_table[row] + transmission_table[row][char_list[0]]
        except KeyError:  # never observed
            score_matrix[row, 0] = zero
            # the rest chars
    for col in range(1, len(char_list)):
        for row in range(4):
            max_score = zero
            track = 0
            for i in range(4):
                this_score = score_matrix[i, col - 1]
                if this_score == zero:
                    continue
                if transmission_table[row].__contains__(char_list[col]):
                    this_score += transition_table[i, row] + transmission_table[row][char_list[col]]
                    if this_score < zero:
                        raise Exception(this_score)
                        # continue
                else:
                    continue
                if this_score > max_score:
                    max_score = this_score
                    track = i
            score_matrix[row, col] = max_score
            track_matrix[row, col] = track
    # backward
    predicted_tag = ''
    tag_index = int(np.argmax(score_matrix[:, length - 1]))
    for col in range(score_matrix.shape[1] - 1, 0, -1):
        predicted_tag = tags[tag_index] + predicted_tag
        # previous
        tag_index = int(track_matrix[tag_index, col])
    return list(tags[tag_index] + predicted_tag)


def viterbi_decoding2D(char_list2D):
    predicted_tag_list2D = []
    for char_list in char_list2D:
        predicted_tag_list2D.append(viterbi_decoding(char_list))
    return predicted_tag_list2D


def calculate_accuracy(tag_list2D, predicted_tag_list2D):
    total = 0
    right = 0
    for tag_list, predicted_tag_list in zip(tag_list2D, predicted_tag_list2D):
        for tag, predicted_tag in zip(tag_list, predicted_tag_list):
            total += 1
            if tag == predicted_tag:
                right += 1
    return right / total


def export_model(filepath='hmm_model.json'):
    model = {
        'init_table': init_table,
        'transmission_table': transmission_table,
        'transition_table': transmission_table
    }
    json.dump(model, open(filepath, 'w'))


def import_model(filepath='hmm_model.json'):
    model = json.load(open(filepath))
    init_table = model['init_table']
    transmission_table = model['transmission_table']
    transition_table = model['transition_table']
    return init_table, transmission_table, transition_table


if __name__ == '__main__':
    init_table, transmission_table, transition_table = import_model()
    char_list2D, tag_list2D = get_corpus(filepath='../data/train.utf8')
    # init_table = get_init_table(tag_list2D)
    # transmission_table = get_transmission_table(char_list2D, tag_list2D)
    # transition_table = get_transition_table(tag_list2D)
    # export_model()
    predicted_tag_list2D = viterbi_decoding2D(char_list2D)
    print(calculate_accuracy(tag_list2D, predicted_tag_list2D))


