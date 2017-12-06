# encoding=utf-8
import re
import tool.file_tool as file_tool


class TemplateReader:
    def __init__(self):
        self.macro = re.compile(r'%x\[(?P<row>[\d-]+),(?P<col>[\d]+)\]')
        self.xy_split_char = ','
        self.macro_split_char = '/'
        self.id_macro_split_char = ':'

        self.all_macros = []
        self.template_id_list = []

    def get_template_id_list(self):
        return self.template_id_list

    def get_all_macros(self):
        return self.all_macros

    def read(self, template_path):
        status, response = file_tool.read_file_lines(template_path)
        if not status:
            print response
            return False, response
        lines = response

        template_list = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('U') or line.startswith('B'):
                template_list.append(line)

        status, response = self.__get_macro_list(template_list)
        if not status:
            print "Error in get_macros_list"
            return False, response
        macros_list = response
        self.all_macros = macros_list

        return True, ''

    def __replace(self, m):
        row = m.group('row')
        col = m.group('col')
        return row + self.xy_split_char + col

    def __get_macro_list(self, template_line_list):
        macros_list = []
        for template_line in template_line_list:
            # 对line进行replace
            replaced_string = re.sub(self.macro, self.__replace, template_line)
            split_array = replaced_string.split(self.id_macro_split_char)
            if len(split_array) != 2:
                print "Error template_line [%s] split by : not equal 2 " % template_line
                continue
            # 获取这个template的字符串信息
            template_id = split_array[0]
            template_coordinates = split_array[1]
            self.template_id_list.append(template_id)

            template_xy_list = []
            coordinate_list = template_coordinates.split(self.macro_split_char)
            for coordinate in coordinate_list:
                xy_list = coordinate.split(self.xy_split_char)
                if len(xy_list) != 2:
                    print "Error coordinate [%s] split by , not equal 2 " % coordinate
                    continue
                x = int(xy_list[0])
                y = int(xy_list[1])
                template_xy_list.append([x, y])
            # 一个template对应的输入macro,包含id和x,y坐标list
            one_macro = {
                "id": template_id,
                "xy_list": template_xy_list,
            }
            macros_list.append(one_macro)

        return True, macros_list


if __name__ == "__main__":
    template_file = "data/template.utf8"
    x = TemplateReader()
    x.read(template_file)
    print x.get_all_macros()
