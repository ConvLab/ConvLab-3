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
        data_path = download_unified_datasets('camrest', 'data.zip', os.path.dirname(os.path.abspath(__file__)))
        archive = ZipFile(data_path)
        self.dbs = {}
        with archive.open('data/CamRestDB.json') as f:
            self.dbs['restaurant'] = json.loads(f.read())
        self.slot2dbattr = {
            'price range': 'pricerange',
        }

    def query(self, domain: str, state: dict, topk: int, ignore_open=False, soft_contraints=(), fuzzy_match_ratio=60) -> list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""
        # query the db
        assert domain == 'restaurant'
        state = list(map(lambda ele: (self.slot2dbattr.get(ele[0], ele[0]), ele[1]) if not(ele[0] == 'area' and ele[1] == 'center') else ('area', 'centre'), state))

        found = []
        for i, record in enumerate(self.dbs[domain]):
            constraints_iterator = zip(state, [False] * len(state))
            soft_contraints_iterator = zip(soft_contraints, [True] * len(soft_contraints))
            for (key, val), fuzzy_match in chain(constraints_iterator, soft_contraints_iterator):
                if val in ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care"]:
                    pass
                else:
                    try:
                        if key not in record:
                            continue
                        if record[key].strip() == '?':
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
    res = db.query("restaurant", [['price range', 'expensive']], topk=3)
    print(res, len(res))
    # print(db.query("hotel", [['price range', 'moderate'], ['stars','4'], ['type', 'guesthouse'], ['internet', 'yes'], ['parking', 'no'], ['area', 'east']]))
