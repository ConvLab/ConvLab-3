# -*- coding: utf-8 -*-
import sys
import numpy as np
from convlab.util.multiwoz.lexicalize import delexicalize_da, flat_da
from .vector_binary import VectorBinary


class VectorBinaryFuzzy(VectorBinary):

    def __init__(self, dataset_name='multiwoz21', character='sys', use_masking=False, manually_add_entity_names=True,
                 seed=0):

        super().__init__(dataset_name, character, use_masking, manually_add_entity_names, seed)

    def dbquery_domain(self, domain):
        """
        query entities of specified domain
        Args:
            domain string:
                domain to query
        Returns:
            entities list:
                list of entities of the specified domain
        """
        # Get all user constraints
        constraints = [[slot, value] for slot, value in self.state[domain].items() if value] \
            if domain in self.state else []
        xx = self.db.query(domain=domain, state=[], soft_contraints=constraints, fuzzy_match_ratio=100, topk=10)
        yy = self.db.query(domain=domain, state=constraints, topk=10)
        #print("STRICT:", yy)
        #print("FUZZY :", xx)
        #if len(yy) == 1 and len(xx) > 1:
        #    import pdb
        #    pdb.set_trace()
        return xx
        #return self.db.query(domain=domain, state=[], soft_contraints=constraints, fuzzy_match_ratio=100, topk=10)
