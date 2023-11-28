import json
import os
import random
from fuzzywuzzy import fuzz
from itertools import chain
from zipfile import ZipFile
from copy import deepcopy
from convlab.util.unified_datasets_util import BaseDatabase, download_unified_datasets


class Database(BaseDatabase):
    def __init__(self):
        """extract data.zip and load the database."""
        data_path = download_unified_datasets('multiwoz21', 'data.zip', os.path.dirname(os.path.abspath(__file__)))
        archive = ZipFile(data_path)
        self.domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'police']
        self.dbs = {}
        for domain in self.domains:
            with archive.open('data/{}_db.json'.format(domain)) as f:
                self.dbs[domain] = json.loads(f.read())
        # add some missing information
        self.dbs['taxi'] = {
            "taxi_colors": ["black","white","red","yellow","blue","grey"],
            "taxi_types":  ["toyota","skoda","bmw","honda","ford","audi","lexus","volvo","volkswagen","tesla"],
            "taxi_phone": ["^[0-9]{10}$"]
        }
        self.dbs['police'][0]['postcode'] = "cb11jg"
        for entity in self.dbs['hospital']:
            entity['postcode'] = "cb20qq"
            entity['address'] = "Hills Rd, Cambridge"

        self.slot2dbattr = {
            'open hours': 'openhours',
            'price range': 'pricerange',
            'arrive by': 'arriveBy',
            'leave at': 'leaveAt',
            'train id': 'trainID'
        }

    def query(self, domain: str, state: dict, topk: int, ignore_open=False, soft_contraints=(), fuzzy_match_ratio=60) -> list:
        """
        return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state.
        :param state: support two formats: 1) [[slot,value], [slot,value]...]; 2) {domain: {slot: value, slot: value...}} (the same as belief state)
        """
        if isinstance(state, dict):
            assert domain in state, print(f"domain {domain} not in state {state}")
            state = state[domain].items()
        # query the db
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
            'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
            'taxi_phone': ''.join([str(random.randint(1, 9)) for _ in range(11)])}]
        if domain == 'police':
            return deepcopy(self.dbs['police'])
        if domain == 'hospital':
            department = None
            for key, val in state:
                if key == 'department':
                    department = val
            if not department:
                for key, val in soft_contraints:
                    if key == 'department':
                        department = val
            if not department:
                return deepcopy(self.dbs['hospital'])
            else:
                return [deepcopy(x) for x in self.dbs['hospital'] if x['department'].lower() == department.strip().lower()]
        state = list(map(lambda ele: (self.slot2dbattr.get(ele[0], ele[0]), ele[1]) if not(ele[0] == 'area' and ele[1] == 'center') else ('area', 'centre'), state))
        soft_contraints = list(map(lambda ele: (self.slot2dbattr.get(ele[0], ele[0]), ele[1]) if not(ele[0] == 'area' and ele[1] == 'center') else ('area', 'centre'), soft_contraints))

        found = []
        for i, record in enumerate(self.dbs[domain]):
            constraints_iterator = zip(state, [False] * len(state))
            soft_contraints_iterator = zip(soft_contraints, [True] * len(soft_contraints))
            for (key, val), fuzzy_match in chain(constraints_iterator, soft_contraints_iterator):
                if val in ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care", "do not care"]:
                    pass
                else:
                    try:
                        if key not in record:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        # elif ignore_open and key in ['destination', 'departure', 'name']:
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        elif record[key].strip() == '?':
                            # '?' matches any constraint
                            continue
                        else:
                            if not fuzzy_match:
                                if val.strip().lower() != record[key].strip().lower():
                                    break
                            else:
                                if fuzz.partial_ratio(val.strip().lower(), record[key].strip().lower()) < fuzzy_match_ratio:
                                    break
                    except:
                        continue
            else:
                res = deepcopy(record)
                res['Ref'] = '{0:08d}'.format(i)
                found.append(res)
                if len(found) == topk:
                    return found
        return found


if __name__ == '__main__':
    db = Database()
    assert issubclass(Database, BaseDatabase)
    assert isinstance(db, BaseDatabase)
    # both calls will be ok
    res1 = db.query("restaurant", [['price range', 'expensive']], topk=3)
    res2 = db.query("restaurant", {'restaurant':{'price range': 'expensive'}}, topk=3)
    assert res1 == res2
    print(res1, len(res1))
    # print(db.query("hotel", [['price range', 'moderate'], ['stars','4'], ['type', 'guesthouse'], ['internet', 'yes'], ['parking', 'no'], ['area', 'east']]))
