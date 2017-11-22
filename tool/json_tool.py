# encoding=utf-8
import json


def json_dump(json_obj):
    print json.dumps(json_obj, indent=4, sort_keys=True, ensure_ascii=False)
