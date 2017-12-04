# encoding=utf-8
import crf_train as crf_train
import feature_function as feature_func

def test_prediction(template_file, sentence, correct_states_string):
    pass


def test_qianqi():
    # 1 测试训练好的模型
    template_file = "data/template_small"
    sentence = u"北京天气"
    correct_states_string = "BEBE"
    test_prediction(template_file, sentence, correct_states_string)


def test_verbit():
    training_set = "data/train_beijing"
    template = "data/template_beijing"
    model_output = "data/crf_model_output_file"
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


if __name__ == "__main__":
    test_verbit()
