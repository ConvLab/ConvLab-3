"""Modifed from https://github.com/thu-coai/CrossWOZ/blob/master/convlab2/util/crosswoz/dbquery.py"""
import json
import os
import re
from zipfile import ZipFile

from convlab.util.unified_datasets_util import (BaseDatabase,
                                                download_unified_datasets)


def contains(arr, s):
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))

class Database(BaseDatabase):
    def __init__(self):
        """extract data.zip and load the database."""
        data_path = download_unified_datasets('crosswoz', 'data.zip', os.path.dirname(os.path.abspath(__file__)))
        archive = ZipFile(data_path)
        self.domains = ['景点', '餐馆', '酒店', '地铁', '出租']
        domain2eng = {'景点': 'attraction', '餐馆': 'restaurant', '酒店': 'hotel', '地铁': 'metro'}
        self.data = {}
        for domain in self.domains[:-1]:
            with archive.open('data/{}_db.json'.format(domain2eng[domain])) as f:
                self.data[domain] = json.loads(f.read())
        
        self.schema = {
            '景点': {
                '名称': {'params': None},
                '门票': {'type': 'between', 'params': [None, None]},
                '游玩时间': {'params': None},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None},
            },
            '餐馆': {
                '名称': {'params': None},
                '推荐菜': {'type': 'multiple_in', 'params': None},
                '人均消费': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '酒店': {
                '名称': {'params': None},
                '酒店类型': {'params': None},
                '酒店设施': {'type': 'multiple_in', 'params': None},
                '价格': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '地铁': {
                '起点': {'params': None},
                '终点': {'params': None},
            },
            '出租': {
                '起点': {'params': None},
                '终点': {'params': None},
            }
        }

    def query(self, domain: str, state: dict, topk: int) -> list:
        """
        return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state.
        query database using belief state, return list of entities, same format as database
        :param state: belief state of the format {domain: {slot: value}}
        :param domain: maintain by DST, current query domain
        :param topk: max number of entities
        :return: list of entities
        """
        if not domain:
            return []
        cur_query_form = {}
        for slot, value in state[domain].items():
            if not value:
                continue
            if slot == '出发地':
                slot = '起点'
            elif slot == '目的地':
                slot = '终点'
            if slot == '名称':
                # DONE: if name is specified, ignore other constraints
                cur_query_form = {'名称': value}
                break
            elif slot == '评分':
                if re.match('(\d\.\d|\d)', value):
                    if re.match('\d\.\d', value):
                        score = float(re.match('\d\.\d', value)[0])
                    else:
                        score = int(re.match('\d', value)[0])
                    cur_query_form[slot] = [score, None]
                # else:
                #     assert 0, value
            elif slot in ['门票', '人均消费', '价格']:
                low, high = None, None
                if re.match('(\d+)-(\d+)', value):
                    low = int(re.match('(\d+)-(\d+)', value)[1])
                    high = int(re.match('(\d+)-(\d+)', value)[2])
                elif re.match('\d+', value):
                    if '以上' in value:
                        low = int(re.match('\d+', value)[0])
                    elif '以下' in value:
                        high = int(re.match('\d+', value)[0])
                    else:
                        low = high = int(re.match('\d+', value)[0])
                elif slot == '门票':
                    if value == '免费':
                        low = high = 0
                    elif value == '不免费':
                        low = 1
                    else:
                        print(value)
                        # assert 0
                cur_query_form[slot] = [low, high]
            else:
                cur_query_form[slot] = value
        cur_res = self.query_schema(field=domain, args=cur_query_form)
        if domain == '出租':
            res = [cur_res]
        elif domain == '地铁':
            res = []
            for r in cur_res:
                if not res and '起点' in r[0]:
                    res.append(r)
                    break
            for r in cur_res:
                if '终点' in r[0]:
                    res.append(r)
                    break
        else:
            res = cur_res

        return res[:topk]
    
    def query_schema(self, field, args):
        if not field in self.schema:
            raise Exception('Unknown field %s' % field)
        if not isinstance(args, dict):
            raise Exception('`args` must be dict')
        db = self.data.get(field)
        plan = self.schema[field]
        for key, value in args.items():
            if not key in plan:
                raise Exception('Unknown key %s' % key)
            value_type = plan[key].get('type')
            if value_type == 'between':
                if not value[0] is None:
                    plan[key]['params'][0] = float(value[0])
                if not value[1] is None:
                    plan[key]['params'][1] = float(value[1])
            else:
                if not isinstance(value, str):
                    raise Exception('Value for `%s` must be string' % key)
                plan[key]['params'] = value
        if field in ['地铁', '出租']:
            s = plan['起点']['params']
            e = plan['终点']['params']
            if not s or not e:
                return []
            if field == '出租':
                return [
                    '出租 (%s - %s)' % (s, e), {
                        '领域': '出租',
                        '车型': '#CX',
                        '车牌': '#CP'
                    }
                ]
            else:
                def func1(item):
                    if item[0].find(s) >= 0:
                        return ['(起点) %s' % item[0], item[1]]

                def func2(item):
                    if item[0].find(e) >= 0:
                        return ['(终点) %s' % item[0], item[1]]
                    return None

                return list(filter(lambda item: not item is None, list(map(func1, db)))) + list(
                    filter(lambda item: not item is None, list(map(func2, db))))

        def func3(item):
            details = item[1]
            for key, val in args.items():
                val = details.get(key)
                absence = val is None
                options = plan[key]
                if options.get('type') == 'between':
                    L = options['params'][0]
                    R = options['params'][1]
                    if not L is None:
                        if absence:
                            return False
                    else:
                        L = float('-inf')
                    if not R is None:
                        if absence:
                            return False
                    else:
                        R = float('inf')
                    if L > val or val > R:
                        return False
                elif options.get('type') == 'in':
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if contains(val, s):
                            return False
                elif options.get('type') == 'multiple_in':
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        sarr = list(filter(lambda t: bool(t), s.split(' ')))
                        if len(list(filter(lambda t: contains(val, t), sarr))):
                            return False
                else:
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if val.find(s) < 0:
                            return False
            return True

        return [x[1] for x in filter(func3, db)]


if __name__ == '__main__':
    db = Database()
    assert issubclass(Database, BaseDatabase)
    assert isinstance(db, BaseDatabase)
    res = db.query("餐馆", {"餐馆":{'评分':'4.5以上'}}, topk=3)
    from pprint import pprint
    pprint(res)
    print(len(res))
