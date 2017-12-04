# encoding=utf-8


class FeatureFunction:
    def __init__(self, params):
        self.prev_label_key = "prev_label"
        self.label_key = "label"
        self.feature_key = "feature"
        self.macro_key = "macro_dict"

        self.func_params = ''
        self.function_type = ''

        if params.has_key(self.prev_label_key) and params.has_key(self.label_key) \
                and params.has_key(self.feature_key) and params.has_key(self.macro_key):
            # 初始化参数信息
            self.func_params = {
                self.prev_label_key: params[self.prev_label_key],
                self.label_key: params[self.label_key],
                self.feature_key: params[self.feature_key]
            }
            # 一个特征函数对应的模板,对应的输入macro,例如%x[0,0]/%x[0,1],包含id与xy_list
            self.macro_dict = params[self.macro_key]
        else:
            raise ValueError("invalid params, must have at least prev_label, label_key feature_key and macro_dict!")

    def calculate(self, prev_label, label, feature):
        is_prev_label_true = self.func_params[self.prev_label_key] == '' \
                             or self.func_params[self.prev_label_key] == prev_label
        is_label_true = self.func_params[self.label_key] == '' or self.func_params[self.label_key] == label
        is_feature_true = self.func_params[self.feature_key] == '' or self.func_params[self.feature_key] == feature
        if is_prev_label_true and is_label_true and is_feature_true:
            return 1
        else:
            return 0

    def get_macro_dict(self):
        if self.macro_dict != '':
            return self.macro_dict
        else:
            return {}

    def to_string(self):
        return "func = if (prev_label = [%s], label = [%s] and feature = [%s] ) return 1 else return 0  | macro = [%s]" \
               % (self.func_params[self.prev_label_key], self.func_params[self.label_key],
                  self.func_params[self.feature_key], self.macro_dict)


if __name__ == "__main__":
    params = {
        "prev_label": '',
        "label": 'B',
        "feature": 'beijing'
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
        "feature": 'beijing'
    }
    myBigramFunc = FeatureFunction(params)
    print myBigramFunc.calculate('B', 'E', 'beijing')
    print myBigramFunc.calculate('B', 'C', 'beijing')

    params = {
        "prev_label": 'B',
        "label": 'E',
        "feature": ''
    }
    myFeatureFunc = FeatureFunction(params)
    print myFeatureFunc.calculate('B', 'E', '')
    print myFeatureFunc.calculate('B', 'C', '')
