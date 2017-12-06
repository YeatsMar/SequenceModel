# encoding=utf-8


# 给定一对输入,一列特征函数和对应参数,返回输入对所有这些特征函数求值的和
def get_feature_func_sum(feature_func_list, func_param_vector, pre_label, this_label, this_feature):
    feature_func_num = len(feature_func_list)
    feature_func_sum = 0.0
    # 遍历所有的特征函数
    for k in range(feature_func_num):
        func = feature_func_list[k]
        func_param = func_param_vector[k]
        # 计算一个特征函数的结果
        func_result = func_param * func.calculate(pre_label, this_label, this_feature)
        feature_func_sum += func_result
    return feature_func_sum


class FeatureFunction:
    def __init__(self, params):
        # 参数矩阵中必须存在的键,如果键对应的值不需要（匹配任意）,就输入空字符串
        self.prev_label_key = "prev_label"
        self.label_key = "label"
        self.feature_key = "feature"
        self.macro_key = "macro_dict"

        self.func_params = ''
        self.function_type = ''

        # 三个参数,prev_label,label,feature必须在键中存在,如果为空则代表任意都可以匹配
        # 例如状态转移矩阵对应的feature就是空,只有前后两个label的转换
        if params.has_key(self.prev_label_key) and params.has_key(self.label_key) \
                and params.has_key(self.feature_key) and params.has_key(self.macro_key):
            # 初始化参数信息
            self.func_params = {
                self.prev_label_key: params[self.prev_label_key],
                self.label_key: params[self.label_key],
                self.feature_key: params[self.feature_key]
            }
            # 第四个输入参数中的键:一个特征函数对应的模板,对应的输入macro,例如%x[0,0]/%x[0,1],包含id与xy_list
            self.macro_dict = params[self.macro_key]
        else:
            raise ValueError("invalid params, must have at least prev_label, label_key feature_key and macro_dict!")

    # 根据输入的3个参数,和本函数的参数信息,计算结果
    def calculate(self, prev_label, label, feature):
        is_prev_label_true = self.func_params[self.prev_label_key] == '' \
                             or self.func_params[self.prev_label_key] == prev_label
        is_label_true = self.func_params[self.label_key] == '' or self.func_params[self.label_key] == label
        is_feature_true = self.func_params[self.feature_key] == '' or self.func_params[self.feature_key] == feature
        # 必须3个参数对应的值都相同才能返回1
        if is_prev_label_true and is_label_true and is_feature_true:
            return 1
        else:
            return 0

    # 返回这个函数对应的macro信息（从特征模板里获取的）
    def get_macro_dict(self):
        if self.macro_dict != '':
            return self.macro_dict
        else:
            return {}

    def to_string(self):
        return "prev_l = [%s]\tl = [%s]\t feature = [%s]\tmacro = [%s]" \
               % (self.func_params[self.prev_label_key], self.func_params[self.label_key],
                  self.func_params[self.feature_key], self.macro_dict)

    def to_postitive_string(self):
        if not self.macro_dict:
            return 'Trans:' + self.func_params[self.prev_label_key] + '-->' + self.func_params[self.label_key] + \
                   '\t' + self.func_params[self.feature_key]
        return str(self.macro_dict['id']) + '\t' + self.func_params[self.feature_key]


if __name__ == "__main__":
    params = {
        "prev_label": '',
        "label": 'B',
        "feature": 'beijing',
        "macro_dict": {}
    }
    myUnigramFunc = FeatureFunction(params)
    # unigram只计算label与feature 不关心第一个,因为prev_label是空的
    print myUnigramFunc.calculate('', 'B', 'beijing')
    print myUnigramFunc.calculate('xx', 'B', 'beijing')
    print myUnigramFunc.calculate('', 'C', 'beijing')
    print myUnigramFunc.calculate('xx', 'C', 'beijing')

    params = {
        "prev_label": 'B',
        "label": 'E',
        "feature": 'beijing',
        "macro_dict": {}
    }
    myBigramFunc = FeatureFunction(params)
    print myBigramFunc.calculate('B', 'E', 'beijing')
    print myBigramFunc.calculate('B', 'C', 'beijing')

    params = {
        "prev_label": 'B',
        "label": 'E',
        "feature": '',
        "macro_dict": {}
    }
    myFeatureFunc = FeatureFunction(params)
    print myFeatureFunc.calculate('B', 'E', '')
    print myFeatureFunc.calculate('B', 'C', '')
