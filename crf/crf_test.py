# encoding=utf-8
import crf_train as crf_train
import feature_function as feature_func
import ai_lab_2.tool.file_tool as file_tool
import ai_lab_2.tool.model_tool as model_tool
import viterbi_function as viterbi_func
import model_file_io as model_file_io
import ai_lab_2.tool.global_variable as global_variable


def test_verbit():
    training_set = "/Users/kylin/Downloads/normandy_scripts/train_beijing"
    template = "/Users/kylin/Downloads/normandy_scripts/template_beijing"
    model_output = "/Users/kylin/Downloads/normandy_scripts/crf_model_output_file"
    crf = crf_train.CRF_TRAINER(training_set, template, model_output)
    # test the case found in zhihu

    # implement the case in https://zhuanlan.zhihu.com/p/29989121
    # input 1, feature function list
    feature_func_list = []
    params = {"prev_label": 'n', "label": 'v', "feature": 'b'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": 'n', "label": 'v', "feature": 'c'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": 'n', "label": 'n', "feature": 'b'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": 'v', "label": 'n', "feature": 'c'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": 'v', "label": 'n', "feature": 'b'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": 'v', "label": 'v', "feature": 'c'}
    feature_func_list.append(feature_func.FeatureFunction(params))

    params = {"prev_label": '', "label": 'n', "feature": 'a'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": '', "label": 'v', "feature": 'a'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": '', "label": 'v', "feature": 'b'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": '', "label": 'n', "feature": 'b'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": '', "label": 'n', "feature": 'c'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    params = {"prev_label": '', "label": 'v', "feature": 'c'}
    feature_func_list.append(feature_func.FeatureFunction(params))
    # input 2, feature function param vector
    func_param_vector = [1, 1, 0.5, 1, 1, 0.2, 1, 0.5, 0.5, 0.8, 0.8, 0.5]
    if len(func_param_vector) != len(feature_func_list):
        print "init error"
        return False, ''
    # input 3, label result array
    label_array = ['n', 'v']
    # input 4, input sentence to be labeled
    input_sentence = "abc"
    crf.verbit(feature_func_list, func_param_vector, label_array, input_sentence)


def test_beijing():
    # 0 模型的基本输入
    base_dir = "/Users/kylin/Desktop/ai_lab_my/"
    valid_tag_list = global_variable.state_array

    # 训练模型输入1: A training set of tagged sentences
    training_file = base_dir + "train_beijing"
    # training_file = base_dir + "train.utf8"
    # training_file = base_dir + "train.tiny"
    # 训练模型输入2: 迭代的最大次数
    parameter_T = 100
    # 训练模型输入3: template文件-->决定了所有特征函数
    template = base_dir + "template_beijing"

    # model init
    crf_model = crf_train.CRF_TRAINER(training_file, template, valid_tag_list)
    # train model with sentence and valid states for T times
    status, feature_func_obj_list, func_param_vector = crf_model.train_model(parameter_T)
    if not status:
        print "Error in train_model"
        return False, ''

    # 模型训练输出:特征函数和相应的权重
    model_output = base_dir + "crf_model_output_file"
    status, response = model_file_io.write_model_to_file(feature_func_obj_list, func_param_vector, model_output)
    if not status:
        print "Error in write_model_to_file"
        return False, ''
    model_output = base_dir + "crf_model_positive_file"
    status, response = model_file_io.write_positive_func_to_file(feature_func_obj_list, func_param_vector, model_output)
    if not status:
        print "Error in write_model_to_file"
        return False, ''

    # 获取输入文件的所有字符和label字符串
    test_file = training_file
    status, sentences, label_of_sentences = file_tool.read_training_sentences(test_file)
    if not status:
        print "Error when read_training_sentences"
        return False, ''
    print "sentence count = [%d]" % len(sentences)

    # 无视换行,看做一个大句子,方便统计正确率
    status, all_char_list, all_correct_state = file_tool.read_training_file(training_file)
    if not status:
        print "Error in read_training_file"
        return False, ''

    # use model to predict sentence tags
    all_returned_state = ""
    for sentence in sentences:
        status, result = viterbi_func.verbit(feature_func_obj_list, func_param_vector, valid_tag_list, sentence)
        if not status:
            print "Error in verbit"
            return False, ''
        all_returned_state += "".join(result)

    all_correct_state_str = "".join(all_correct_state)

    # print accuracy
    print "####### print result ######## "
    st, resp = model_tool.print_predict_result(all_returned_state, all_correct_state_str, all_char_list)
    if not st:
        print resp
        return False, ''
    print "####### TEST END ######## "


if __name__ == "__main__":
    test_beijing()
