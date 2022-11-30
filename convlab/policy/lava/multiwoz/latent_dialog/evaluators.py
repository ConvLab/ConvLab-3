from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
import numpy as np
import convlab.policy.lava.multiwoz.latent_dialog.normalizer.delexicalize as delex
from convlab.policy.lava.multiwoz.latent_dialog.utils import get_tokenize, get_detokenize
from collections import Counter, defaultdict
from nltk.util import ngrams
from convlab.policy.lava.multiwoz.latent_dialog.corpora import SYS, USR, BOS, EOS
from sklearn.feature_extraction.text import CountVectorizer
import json
from convlab.policy.lava.multiwoz.latent_dialog.normalizer.delexicalize import normalize
import sqlite3
import os
import random
import logging
import pdb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from scipy.stats import gmean



class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class BleuEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.labels = list()
        self.hyps = list()

    def initialize(self):
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def get_report(self):
        tokenize = get_tokenize()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        for label, hyp in zip(self.labels, self.hyps):
            # label = label.replace(EOS, '')
            # hyp = hyp.replace(EOS, '')
            # ref_tokens = tokenize(label)[1:]
            # hyp_tokens = tokenize(hyp)[1:]
            ref_tokens = tokenize(label)
            hyp_tokens = tokenize(hyp)
            refs.append([ref_tokens])
            hyps.append(hyp_tokens)
        bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        report = '\n===== BLEU = %f =====\n' % (bleu,)
        return '\n===== REPORT FOR DATASET {} ====={}'.format(self.data_name, report)


class MultiWozDB(object):
    # loading databases
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__).replace('latent_dialog', '')

    for domain in domains:
        db = os.path.join(CUR_DIR, 'data/norm-multi-woz/db/{}-dbase.db'.format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    def queryResultVenues(self, domain, turn, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        flag = True
        for key, val in items:
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            return self.dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    CUR_DIR = os.path.dirname(__file__).replace('latent_dialog', '')
    logger = logging.getLogger()
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = delex.prepareSlotValuesIndependent()
        self.delex_dialogues = json.load(open(os.path.join(self.CUR_DIR, 'data/norm-multi-woz/delex.json')))
        self.db = MultiWozDB()
        self.labels = list()
        self.hyps = list()

    def initialize(self):
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
        # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
            # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, sent_t in enumerate(dialog):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t: # undo delexicalization if system generates [domain_name] or [domain_id]
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']: 
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        # in this case, look for the actual offered venues based on true belief state
                        venues = self.db.queryResultVenues(domain, realDialogue['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if '[' + domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # assumption doesn't always hold, maybe it's better if name is provided by user that it is ignored?
            if 'info' in realDialogue['goal'][domain]:
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            if domain == 'train':
                if not venue_offered[domain]:
                    # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
                    if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
                        venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH (between offered venue by generated dialogue and venue actually fitting to the criteria)
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                # if venue offered is not dict
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]: # yields false positive, does not match what is offered with real dialogue?
                    match += 1
                    match_stat = 1
                # if venue offered is dict
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues: # actually checks the offered venue
                    match += 1
                    match_stat = 1
            # other domains
            else:
                if domain + '_name]' in venue_offered[domain]: # yields false positive, in terms of occurence and correctness
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            # only count success if all domain has matches
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS (whether the requestable info in realDialogue is generated by the system)
        # if no match, then success is assumed to be 0
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0: # if there is no requestable, assume to be succesful. incorrect, cause does not count false positives. 
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats

    def _evaluateGeneratedDialogue_new(self, dialog, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, sent_t in enumerate(dialog):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, realDialogue['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
            #         if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # Wrong one in HDSA
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue from corpus.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, dialog['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                                    # return goal, 0, match, real_requestables
                            elif 'train_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            # if dialog['goal'][domain].has_key('info'):
            if 'info' in dialog['goal'][domain]:
                # if dialog['goal'][domain]['info'].has_key('name'):
                if 'name' in dialog['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        # HARD (0-1) EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match, success = 0, 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
                # print(goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1

            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, stats

    def _evaluateRolloutDialogue(self, dialog):
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or domain+'_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        venue_offered[domain] = '[' + domain + '_name]'
                        """
                        venues = self.db.queryResultVenues(domain, dialog['log'][t * 2 + 1])
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                        """
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if True or dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if True or dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')
                                    # return goal, 0, match, real_requestables
                            elif 'train_reference' in sent_t:
                                if True or dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            # if dialog['goal'][domain].has_key('info'):
            if 'info' in dialog['goal'][domain]:
                # if dialog['goal'][domain]['info'].has_key('name'):
                if 'name' in dialog['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        # REWARD CALCULATION
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        match, success = 0.0, 0.0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        match = min(1.0, float(match) / len(goal.keys()))

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            success = min(1.0, float(success) / len(real_requestables))

        return success, match, stats

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def evaluateModel(self, dialogues, mode='valid', new_version=False):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        # pdb.set_trace()
        successes, matches = 0, 0
        corpus_successes, corpus_matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        for filename, dial in dialogues.items():
            if mode == 'rollout':
                success, match, stats = self._evaluateRolloutDialogue(dial)
            else:
                # data is ground truth, dial is generated
                data = delex_dialogues[filename]
                goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename) # only goal and requestables are kept
                corpus_successes += success
                corpus_matches += match
                if new_version:
                    success, match, stats = self._evaluateGeneratedDialogue_new(dial, goal, data, requestables,
                                                                            soft_acc=mode =='offline_rl')
                else:
                    success, match, stats = self._evaluateGeneratedDialogue(dial, goal, data, requestables,
                                                                            soft_acc=mode =='offline_rl')

            successes += success
            matches += match
            total += 1

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]

        report = ""
        report += '{} Corpus Matches : {:2.2f}%, Groundtruth {} Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100), mode, (corpus_matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%, Groundtruth {} Success : {:2.2f}%'.format(mode, (successes / float(total) * 100), mode, (corpus_successes / float(total) * 100)) + "\n"
        report += 'Total number of dialogues: %s, new version=%s ' % (total, new_version)

        self.logger.info(report)
        return report, successes/float(total), matches/float(total)
    
    def get_report(self):
        tokenize = lambda x: x.split()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        tp, fp, fn = 0, 0, 0
        for label, hyp in zip(self.labels, self.hyps):
            ref_tokens = [BOS] + tokenize(label.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            hyp_tokens = [BOS] + tokenize(hyp.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            refs.append([ref_tokens])
            hyps.append(hyp_tokens)

            ref_entities = self._parse_entities(ref_tokens)
            hyp_entities = self._parse_entities(hyp_tokens)
            tpp, fpp, fnn = self._get_tp_fp_fn(ref_entities, hyp_entities)
            tp += tpp
            fp += fpp
            fn += fnn

        # bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        bleu = BLEUScorer().score(hyps, refs) 
        prec, rec, f1 = self._get_prec_recall(tp, fp, fn)
        report = "\nBLEU score {}\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(bleu, prec, rec, f1)
        return report, bleu, prec, rec, f1

    def get_groundtruth_report(self):
        tokenize = lambda x: x.split()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        tp, fp, fn = 0, 0, 0
        for label, hyp in zip(self.labels, self.hyps):
            ref_tokens = [BOS] + tokenize(label.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            refs.append([ref_tokens])

            ref_entities = self._parse_entities(ref_tokens)
            tpp, fpp, fnn = self._get_tp_fp_fn(ref_entities, ref_entities)
            tp += tpp
            fp += fpp
            fn += fnn

        # bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        # bleu = BLEUScorer().score(refs, refs) 
        prec, rec, f1 = self._get_prec_recall(tp, fp, fn)
        # report = "\nGroundtruth BLEU score {}\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(bleu, prec, rec, f1)
        report = "\nGroundtruth\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(prec, rec, f1)
        return report, 0, prec, rec, f1

class SimDialEvaluator(BaseEvaluator):
    CUR_DIR = os.path.dirname(__file__).replace('latent_dialog', '')
    logger = logging.getLogger()
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = delex.prepareSlotValuesIndependent()
        # self.delex_dialogues = json.load(open(os.path.join(self.CUR_DIR, 'data/norm-multi-woz/delex.json')))
        # self.db = MultiWozDB()
        self.labels = list()
        self.hyps = list()

    def initialize(self):
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def get_report(self):
        tokenize = lambda x: x.split()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        tp, fp, fn = 0, 0, 0
        for label, hyp in zip(self.labels, self.hyps):
            ref_tokens = [BOS] + tokenize(label.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            hyp_tokens = [BOS] + tokenize(hyp.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            refs.append([ref_tokens])
            hyps.append(hyp_tokens)

            ref_entities = self._parse_entities(ref_tokens)
            hyp_entities = self._parse_entities(hyp_tokens)
            tpp, fpp, fnn = self._get_tp_fp_fn(ref_entities, hyp_entities)
            tp += tpp
            fp += fpp
            fn += fnn

        # bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        bleu = BLEUScorer().score(hyps, refs) 
        prec, rec, f1 = self._get_prec_recall(tp, fp, fn)
        report = "\nBLEU score {}\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(bleu, prec, rec, f1)
        return report, bleu, prec, rec, f1

    def get_groundtruth_report(self):
        tokenize = lambda x: x.split()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        tp, fp, fn = 0, 0, 0
        for label, hyp in zip(self.labels, self.hyps):
            ref_tokens = [BOS] + tokenize(label.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            refs.append([ref_tokens])

            ref_entities = self._parse_entities(ref_tokens)
            tpp, fpp, fnn = self._get_tp_fp_fn(ref_entities, ref_entities)
            tp += tpp
            fp += fpp
            fn += fnn

        # bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        # bleu = BLEUScorer().score(refs, refs) 
        prec, rec, f1 = self._get_prec_recall(tp, fp, fn)
        # report = "\nGroundtruth BLEU score {}\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(bleu, prec, rec, f1)
        report = "\nGroundtruth\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(prec, rec, f1)
        return report, 0, prec, rec, f1

class TurnEvaluator(BaseEvaluator):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    CLF = "clf"
    REPRESENTATION = "rep"
    ID2TAG = "id2tag"
    TAG2ID = "tag2id"
    logger = logging.getLogger()

    def __init__(self, data_name, turn_corpus, domain_meta):
        self.data_name = data_name
        # train a dialog act classifier
        domain2ids = defaultdict(list)
        for d_id, d in enumerate(turn_corpus):
            domain2ids[d.domain].append(d_id)
        selected_ids = [v[0:1000] for v in domain2ids.values()]
        corpus = [turn_corpus[idx] for idxs in selected_ids for idx in idxs]

        self.model = self.get_intent_tagger(corpus)

        # get entity value vocabulary
        self.domain_id2ent = self.get_entity_dict_from_meta(domain_meta)

        # Initialize containers
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def get_entity_dict_from_meta(self, domain_meta):
        # get entity value vocabulary
        domain_id2ent = defaultdict(set)
        for domain, meta in domain_meta.items():
            domain_id2ent[domain].add("QUERY")
            domain_id2ent[domain].add("GOALS")
            for slot, vocab in meta.sys_slots.items():
                domain_id2ent[domain].add(slot)
                for v in vocab:
                    domain_id2ent[domain].add(v)

            for slot, vocab in meta.usr_slots.items():
                domain_id2ent[domain].add(slot)
                for v in vocab:
                    domain_id2ent[domain].add(v)

        domain_id2ent = {k: list(v) for k, v in domain_id2ent.items()}
        return domain_id2ent

    def get_entity_dict(self, turn_corpus):
        utt2act = {}
        for msg in turn_corpus:
            utt2act[" ".join(msg.utt[1:-1])] = msg

        detokenize = get_detokenize()
        utt2act = {detokenize(k.split()): v for k, v in utt2act.items()}
        self.logger.info("Compress utt2act from {}->{}".format(len(turn_corpus), len(utt2act)))

        # get entity value vocabulary
        domain_id2ent = defaultdict(set)
        for utt, msg in utt2act.items():
            for act in msg.actions:
                paras = act['parameters']
                intent = act['act']
                if intent == 'inform':
                    for v in paras[0].values():
                        domain_id2ent[msg.domain].add(str(v))
                elif intent == 'query':
                    for v in paras[0].values():
                        domain_id2ent[msg.domain].add(v)
                else:
                    for k, v in paras:
                        if v:
                            domain_id2ent[msg.domain].add(v)
        domain_id2ent = {k: list(v) for k, v in domain_id2ent.items()}
        return domain_id2ent

    def get_intent_tagger(self, corpus):
        """
        :return: train a dialog act tagger for system utterances 
        """
        self.logger.info("Train a new intent tagger")
        all_tags, utts, tags = [], [], []
        de_tknize = get_detokenize()
        for msg in corpus:
            utts.append(de_tknize(msg.utt[1:-1]))
            tags.append([a['act'] for a in msg.actions])
            all_tags.extend([a['act'] for a in msg.actions])

        most_common = Counter(all_tags).most_common()
        self.logger.info(most_common)
        tag_set = [t for t, c, in most_common]
        rev_tag_set = {t: i for i, t in enumerate(tag_set)}

        # create train and test set:
        data_size = len(corpus)
        train_size = int(data_size * 0.7)
        train_utts = utts[0:train_size]
        test_utts = utts[train_size:]

        # create y:
        sparse_y = np.zeros([data_size, len(tag_set)])
        for idx, utt_tags in enumerate(tags):
            for tag in utt_tags:
                sparse_y[idx, rev_tag_set[tag]] = 1
        train_y = sparse_y[0:train_size, :]
        test_y = sparse_y[train_size:, :]

        # train classifier
        representation = CountVectorizer(ngram_range=[1, 2]).fit(train_utts)
        train_x = representation.transform(train_utts)
        test_x = representation.transform(test_utts)

        clf = OneVsRestClassifier(SGDClassifier(loss='hinge', max_iter=10)).fit(train_x, train_y)
        pred_test_y = clf.predict(test_x)

        def print_report(score_name, scores, names):
            for s, n in zip(scores, names):
                self.logger.info("%s: %s -> %f" % (score_name, n, s))

        print_report('F1', metrics.f1_score(test_y, pred_test_y, average=None),
                     tag_set)

        x = representation.transform(utts)
        clf = OneVsRestClassifier(SGDClassifier(loss='hinge', max_iter=20)) \
            .fit(x, sparse_y)

        model_dump = {self.CLF: clf, self.REPRESENTATION: representation,
                      self.ID2TAG: tag_set,
                      self.TAG2ID: rev_tag_set}
        # pkl.dump(model_dump, open("{}.pkl".format(self.data_name), "wb"))
        return model_dump

    def pred_ents(self, sentence, tokenize, domain):
        pred_ents = []
        padded_hyp = "/{}/".format("/".join(tokenize(sentence)))
        for e in self.domain_id2ent[domain]:
            count = padded_hyp.count("/{}/".format(e))
            if domain =='movie' and e == 'I':
                continue
            pred_ents.extend([e] * count)
        return pred_ents

    def pred_acts(self, utts):
        test_x = self.model[self.REPRESENTATION].transform(utts)
        pred_test_y = self.model[self.CLF].predict(test_x)
        pred_tags = []
        for ys in pred_test_y:
            temp = []
            for i in range(len(ys)):
                if ys[i] == 1:
                    temp.append(self.model[self.ID2TAG][i])
            pred_tags.append(temp)
        return pred_tags

    """
    Public Functions
    """
    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False):
        reports = []

        errors = []

        for domain, labels in self.domain_labels.items():
            intent2refs = defaultdict(list)
            intent2hyps = defaultdict(list)

            predictions = self.domain_hyps[domain]
            self.logger.info("Generate report for {} for {} samples".format(domain, len(predictions)))

            # find entity precision, recall and f1
            tp, fp, fn = 0.0, 0.0, 0.0

            # find intent precision recall f1
            itp, ifp, ifn = 0.0, 0.0, 0.0

            # backend accuracy
            btp, bfp, bfn = 0.0, 0.0, 0.0

            # BLEU score
            refs, hyps = [], []

            pred_intents = self.pred_acts(predictions)
            label_intents = self.pred_acts(labels)

            tokenize = get_tokenize()
            bad_predictions = []

            for label, hyp, label_ints, pred_ints in zip(labels, predictions, label_intents, pred_intents):
                refs.append([label.split()])
                hyps.append(hyp.split())

                # pdb.set_trace()

                label_ents = self.pred_ents(label, tokenize, domain)
                pred_ents = self.pred_ents(hyp, tokenize, domain)

                for intent in label_ints:
                    intent2refs[intent].append([label.split()])
                    intent2hyps[intent].append(hyp.split())

                # update the intent
                ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ints, pred_ints)
                itp += ttpp
                ifp += ffpp
                ifn += ffnn

                # entity or KB search
                ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ents, pred_ents)
                if ffpp > 0 or ffnn > 0:
                    bad_predictions.append((label, hyp))

                if "query" in label_ints:
                    btp += ttpp
                    bfp += ffpp
                    bfn += ffnn
                else:
                    tp += ttpp
                    fp += ffpp
                    fn += ffnn

            # compute corpus level scores
            bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
            ent_precision, ent_recall, ent_f1 = self._get_prec_recall(tp, fp, fn)
            int_precision, int_recall, int_f1 = self._get_prec_recall(itp, ifp, ifn)
            back_precision, back_recall, back_f1 = self._get_prec_recall(btp, bfp, bfn)

            # compute BLEU w.r.t intents
            intent_report = []
            for intent in intent2refs.keys():
                i_bleu = bleu_score.corpus_bleu(intent2refs[intent], intent2hyps[intent],
                                                smoothing_function=SmoothingFunction().method1)
                intent_report.append("{}: {}".format(intent, i_bleu))

            intent_report = "\n".join(intent_report)

            # create bad cases
            error = ''
            if include_error:
                error = '\nDomain {} errors\n'.format(domain)
                error += "\n".join(['True: {} ||| Pred: {}'.format(r, h)
                                    for r, h in bad_predictions])
            report = "\nDomain: %s\n" \
                     "Entity precision %f recall %f and f1 %f\n" \
                     "Intent precision %f recall %f and f1 %f\n" \
                     "KB precision %f recall %f and f1 %f\n" \
                     "BLEU %f BEAK %f\n\n%s\n" \
                     % (domain,
                        ent_precision, ent_recall, ent_f1,
                        int_precision, int_recall, int_f1,
                        back_precision, back_recall, back_f1,
                        bleu, gmean([ent_f1, int_f1, back_f1, bleu]),
                        intent_report)
            reports.append(report)
            errors.append(error)

        if include_error:
            return "\n==== REPORT===={error}\n========\n {report}".format(error="========".join(errors),
                                                                          report="========".join(reports))
        else:
            return "\n==== REPORT===={report}".format(report="========".join(reports))
