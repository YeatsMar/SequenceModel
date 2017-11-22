# encoding=utf-8

word_begin_char = u'B'
word_end_char = u'E'
word_middle_char = u'I'
single_char = u'S'
state_array = [word_begin_char, word_end_char, word_middle_char, single_char]
state_count = len(state_array)
b_state_index = state_array.index(word_begin_char)
e_state_index = state_array.index(word_end_char)
m_state_index = state_array.index(word_middle_char)
s_state_index = state_array.index(single_char)


init_state_prob_array = [0.9, 0, 0, 0.1]

min_number = -3.14e+100
