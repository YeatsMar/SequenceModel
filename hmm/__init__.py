import codecs
import numpy as np

B = 0
I = 1
E = 2
S = 3

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

def get_init_table(tag_list2D):
    for tag_list in tag_list2D:
        pass
