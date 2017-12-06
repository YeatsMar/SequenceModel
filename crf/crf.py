import codecs
from pprint import pprint
import numpy as np

tags = ['B', 'I', 'E', 'S']
bias = 2


def get_macros(template='data/template.utf8'):
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


def get_corpus(filepath='data/tiny.utf8'):
    with codecs.open(filepath, encoding='utf8') as fopen:
        lines = fopen.readlines()
        char_list = [line[0] for line in lines]  # todo: contain \n   2D list
        tag_list = [line[2] for line in lines]
        sentence = ''.join(char_list)  # unpolluted
        sentence_tag = ''.join(tag_list)
        char_list.insert(0, '_I-1')
        char_list.insert(0, '_I-2')
        char_list.append('_I+1')
        char_list.append('_I+2')
        tag_list.insert(0, '_T-1')
        tag_list.insert(0, '_T-2')
        tag_list.append('_T+1')
        tag_list.append('_T+2')
        return char_list, tag_list, sentence, sentence_tag
        # todo: sentences = sentence.split('\n')


def generate_feature_functions(macros, char_list):
    global bias
    total_num = 0  # num_params == num_feature_functions
    macros = dict(macros)
    feature_functions = dict()
    for id_macro, relative_pos in macros.items():
        this_macro_group = feature_functions[id_macro] = dict()  # id_macro = 'U00'
        char_set = set()
        for i in range(len(char_list) - bias * 2):
            tmp = ''
            for pos in relative_pos:
                tmp += char_list[i + bias + pos]
            char_set.add(tmp)
        for char in char_set:
            if id_macro[0] == 'U':
                this_macro_group[char] = {'B': 0, 'I': 0, 'E': 0, 'S': 0}  # initialize all parameters as 0
                total_num += 4
            elif id_macro[0] == 'B':  # tag-1 tag0
                this_macro_group[char] = {'B': {'B': 0, 'I': 0, 'E': 0, 'S': 0},
                                          'I': {'B': 0, 'I': 0, 'E': 0, 'S': 0},
                                          'E': {'B': 0, 'I': 0, 'E': 0, 'S': 0},
                                          'S': {'B': 0, 'I': 0, 'E': 0, 'S': 0},
                                          '_T-1': {'B': 0, 'I': 0, 'E': 0, 'S': 0}
                                          # todo: impossible for I & E, can manually set some params
                                          }
                total_num += 16
            else:
                raise Exception('check template.utf8 and modify bias!')
    return feature_functions, total_num


def viterbi_process(feature_functions, macros, total_num_f, char_list, tag_list):
    global tags, bias
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
        for id_macro, observations in feature_functions.items():
            observations = dict(observations)
            relative_pos = list(macros[id_macro])
            o = ''
            for pos in relative_pos:
                observe_index = first_char_index + pos
                o += char_list[observe_index]
            if o not in observations:
                continue
            param = observations[o]
            if id_macro[0] == 'U':
                param = param[tag]
            else:  # bigram
                previous_char_index = first_char_index - 1
                pre_tag = tag_list[previous_char_index]
                param = param[pre_tag][tag]
            score += param
        score_matrix[row_score, 0] = score
        row_score += 1
    # for the rest chars
    for this_char_index in range(first_char_index + 1, len(char_list) - bias):  # col
        for row_score in range(len(tags)):
            tag = tags[row_score]
            max_score = 0
            index_track = 0
            for row_track in range(len(tags)):
                pre_tag = tags[row_track]
                score = score_matrix[row_track, this_char_index - 1 - bias]  # sum of previous chars
                for id_macro, observations in feature_functions.items():
                    observations = dict(observations)
                    relative_pos = list(macros[id_macro])
                    o = ''
                    for pos in relative_pos:
                        observe_index = this_char_index + pos
                        o += char_list[observe_index]
                    if o not in observations:
                        continue
                    param = observations[o]
                    if id_macro[0] == 'U':
                        param = param[tag]
                    else:  # bigram
                        param = param[pre_tag][tag]
                    score += param
                if score > max_score:
                    max_score = score
                    index_track = row_track
            track_matrix[row_score, this_char_index - bias] = index_track
            score_matrix[row_score, this_char_index - bias] = max_score
    # backward
    l = len(tag_list) - bias * 2
    l -= 1
    last_column = score_matrix[:, l]
    pre_tag_index = last_column.argmax(0)
    predicted_tag_list= tags[pre_tag_index]
    while l > 0:
        pre_tag_index = int(track_matrix[pre_tag_index, l])
        l -= 1
        predicted_tag_list = tags[pre_tag_index] + predicted_tag_list
    print(tag_list, predicted_tag_list)


if __name__ == '__main__':
    macros = get_macros()
    char_list, tag_list, sentence, sentence_tag = get_corpus()  # the previous 2 are unrelated
    feature_functions, total_num = generate_feature_functions(macros, char_list)
    viterbi_process(feature_functions, macros, total_num, char_list, tag_list)
