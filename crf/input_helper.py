# coding=utf-8
import tool.file_tool as file_tool
import numpy as np


class InputHelper:
    def __init__(self, training_file):
        self.training_file = training_file
        # 输入文件中所有的句子和label
        self.sentences = []
        self.label_of_sentences = []
        # 所有输入字符的大list
        self.input_char_list = []
        self.correct_label_list = []

        # 其他参数
        # 在训练集的第一行之前增加2行不存在的文字,用于处理第一行的特殊情况: %x[-2, 0]
        self.extend_input_index = 2
        self.macro_split_char = '/'
        self.id_macro_split_char = ':'

        # 在输入文件所有字符前后增加2个字符的训练输入矩阵
        self.training_matrix = []
        # 每一个句子对应的训练输入矩阵,缓存提升速度
        self.training_matrix_for_sentence = {}

        # 许多操作要求必须预先读取输入的训练文件
        self.is_training_file_loaded = False

    # 在每一个句子的前后增加了几个无效的label
    # 例如 b-1, b-2, 是为了应对%x[-2]这种情况
    # 如果存在-3,则应返回3
    # TODO 应该从template文件中自动获取,但是感觉template文件并不会怎么调整到-3的情况,-2足够了
    def get_extend_input_index(self):
        return self.extend_input_index

    # 读取训练文件的所有信息
    def read_training_file(self):
        if self.is_training_file_loaded:
            return True, self.sentences, self.label_of_sentences, self.input_char_list, self.correct_label_list
        # 读取训练文件内容1:读取所有句子和对应的标记
        status, sentences, label_of_sentences = file_tool.read_training_sentences(self.training_file)
        if not status:
            print "Error when read_training_sentences"
            return False, ''
        self.sentences = sentences
        self.label_of_sentences = label_of_sentences

        # 读取训练文件内容2:读取所有char和label到一个大list,以便获取所有特征函数
        status, all_char_list, all_state_list = file_tool.read_training_file(self.training_file)
        if not status:
            print "Error in read_training_file"
            return False, ''
        self.input_char_list = all_char_list
        self.correct_label_list = all_state_list
        return True, sentences, label_of_sentences, all_char_list, all_state_list

    # 输入的第一列:char
    # 输入的第二列:label
    # CRF模型可以支持多列,这里只有两列
    # 用来被macro根据下表处理的输入矩阵 training_matrix
    def init_training_matrix(self, input_char_list, input_state_list):
        status, extend_char_list, extend_label_list = self.get_extend_input_array(input_char_list, input_state_list)
        if not status:
            return False, ''
        training_matrix = []
        # 把首尾增加了输入的list加入training_matrix
        for i in range(len(extend_char_list)):
            # x对应哪一行,y对应列,这里只有n行,2列. 第一列是文字,第二列是label
            one_training_example = [extend_char_list[i], extend_label_list[i]]
            training_matrix.append(one_training_example)
        self.training_matrix = training_matrix
        return True, training_matrix

    # 根据macro和当前的位置,获取training_matrix里面需要的feature
    # 例子input_macro = %x[-1,0], current_index(i) = 3, 则获取的feature就是
    # training_matrix里面前一个index(i=2)对应的输入char
    # current_index是加了前后2两个不存在的label之后的下标,最小的是2,所以当x是-2的时候,获取的x就是matrix里面的第0个,就是B:x-2
    def get_input_feature_by_macro(self, input_macro, training_matrix, current_index):
        # 如果input_macro是空,对应的是状态转移特征函数,输入的当前feature是空
        if not input_macro:
            return ''

        # 根据macro在training_matrix里面获取需要的feature
        template_id = input_macro["id"]
        macro_xy_list = input_macro["xy_list"]
        feature_of_macro = []
        for xy_item in macro_xy_list:
            x = xy_item[0] + current_index
            y = xy_item[1]
            # unigram的feature是(当前label+当前输入),例如(B,U02:北)
            # 对应的一个特征函数就是 U02 func1 = if (output = B and feature="U02:北") return 1 else return 0 
            feature_of_macro.append(training_matrix[x][y])
            feature_of_macro.append(self.macro_split_char)
        # 去除末尾的/
        feature_of_macro = feature_of_macro[:-1]

        # U01和U02这些标志位，与特征token组合到一起主要是区分“U01:問”和“U02:問”这类特征
        # 虽然抽取的"字"特征是一样的，但是在CRF中这是有区别的特征。
        feature_for_this_line = template_id + self.id_macro_split_char + "".join(feature_of_macro)
        return feature_for_this_line

    # 在首尾添加输入的拓展训练集
    def get_extend_input_array(self, input_char_list, input_state_list):
        extend_input_char_list = ["_B-2", "_B-1"]
        for input_char in input_char_list:
            extend_input_char_list.append(input_char)
        extend_input_char_list.append("_B+1")
        extend_input_char_list.append("_B+2")

        extend_correct_label_list = ["label_B-2", "label_B-1"]
        for input_state in input_state_list:
            extend_correct_label_list.append(input_state)
        extend_correct_label_list.append("label_B+1")
        extend_correct_label_list.append("label_B+2")

        if len(extend_correct_label_list) != len(input_char_list) + 2 * self.extend_input_index:
            print "get_extend_input_array false"
            return False, '', ''
        return True, extend_input_char_list, extend_correct_label_list

    # 获取一个句子对应的input matrix矩阵（首位拓展两个字符）
    # 例如一个句子是 你好, 返回的矩阵是
    # [b-2, label b-2]
    # [b-1, label b-1]
    # [你, label]
    # [好, label]
    # [b+1, label b+1]
    # [b+2, label b+2]
    # 方便后面迭代的时候,直接获取一个位置的input feature
    # 例子:input_macro = %x[-1,0], input_matrix就是上面的,extend_i_index = 0
    # 获取的就是 b-1,label b-1
    def get_input_matrix_of_sentence(self, sentence_index):
        # 一定要先读取训练集文件
        if not self.is_training_file_loaded:
            self.read_training_file()
            self.is_training_file_loaded = True

        key_for_sentence = str(sentence_index)
        # 已经计算过这个句子对应的输入矩阵,缓存提升速度
        if self.training_matrix_for_sentence.has_key(key_for_sentence):
            return True, self.training_matrix_for_sentence[key_for_sentence]
        # 计算此句子对应的input_matrix
        sentence = self.sentences[sentence_index]
        label_array = self.label_of_sentences[sentence_index]

        # 拓展句子的前后两个字符
        status, extend_char_list, ex_output_list = self.get_extend_input_array(sentence, label_array)
        if not status:
            return False, ''

        # input_matrix有n行,行数是句子长度+4,理论上可以多列,但这里只有两列,一列输入,一列label
        new_input_matrix = np.vstack((extend_char_list, ex_output_list))
        new_input_matrix = np.transpose(new_input_matrix)
        self.training_matrix_for_sentence[key_for_sentence] = new_input_matrix
        return True, new_input_matrix

    # 根据所有的macro获取训练集文件中的所有input feature
    def get_input_macro(self, all_macros, training_set_size):
        # 一定要先读取训练集文件
        if not self.is_training_file_loaded:
            self.read_training_file()
            self.is_training_file_loaded = True

        # 一个macro对应的所有字符串feature,为了后面计算每一行template对应的所有特征函数保留
        template_str_feature_dict = {}
        # 一个tempate对应的所有macro信息
        template_macro_dict = {}
        if training_set_size != len(self.input_char_list):
            print "get_input_macro training_set_size != len(self.input_char_list)"
            return False, ''

        # 遍历一次训练集,获取所有特征函数
        for i in range(training_set_size):
            # 只需要对训练集中每个字符进行macro的匹配, training_matrix是被拓展的, i + extend_input_index
            this_char_index = i + self.extend_input_index
            # 遍历所有模板,获取一个输入（一个字符）对 应的所有特征函数的特征
            for one_macro in all_macros:
                template_id = one_macro.get("id")
                # 获取当前macro定位到的输入feature
                feature_for_this_line = self.get_input_feature_by_macro(one_macro, self.training_matrix,
                                                                        this_char_index)
                # 如果这个模板还不存在feature
                if not template_str_feature_dict.has_key(template_id):
                    template_str_feature_dict[template_id] = set()
                    template_str_feature_dict[template_id].add(feature_for_this_line)
                else:
                    # 如果这个template已经有对应的feature,需要再检查这个feature是否已经存在在已有的feature中
                    # 例子,U02:%x[0,0]会多次匹配重复出现的一个字符,例如l = [B] and feature = [U00:jing]
                    # 这个情况下,feature = [U00:jing]只需要加入一次就可以了,使用set会过滤重复
                    template_str_feature_dict[template_id].add(feature_for_this_line)
                # 一个template对应的一个输入macro
                if not template_macro_dict.has_key(template_id):
                    template_macro_dict[template_id] = one_macro

        return True, template_str_feature_dict, template_macro_dict
