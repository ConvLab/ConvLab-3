import os
import numpy as np
import random
import torch as th
from nltk import RegexpTokenizer
from torch.autograd import Variable
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging
import sys
from collections import defaultdict

INT = 0
LONG = 1
FLOAT = 2


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    @staticmethod
    def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
        pack = Pack()
        for k, v in dictionary.items():
            pack[k] = v
        pack['speaker'] = speaker2id[pack.speaker]
        pack['conf'] = dictionary.get('conf', 1.0)
        utt = pack['utt']
        if 'QUERY' in utt or "RET" in utt:
            utt = str(utt)
            # utt = utt.translate(None, ''.join([':', '"', "{", "}", "]", "["]))
            utt = utt.translate(str.maketrans('', '', ''.join([':', '"', "{", "}", "]", "["])))
            utt = str(utt)
        if include_domain:
            pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
        else:
            pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
        return pack

def get_tokenize():
    return RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize

def get_detokenize():
    return lambda x: TreebankWordDetokenizer().detokenize(x)

def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(th.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(th.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.cuda.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    else:
        if dtype == INT:
            var = var.type(th.IntTensor)
        elif dtype == LONG:
            var = var.type(th.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    return var

def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def set_seed(seed):
    """Sets random seed everywhere."""
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)

def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    #rootLogger.addHandler(consoleHandler)

    if hasattr(config, 'forward_only') and config.forward_only:
        return

    fileHandler = logging.FileHandler(os.path.join(config.saved_path,'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

def get_chat_tokenize():
    return nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize

class missingdict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()

def extract_short_ctx(context, context_lens, backward_size=1):
    utts = []
    for b_id in range(context.shape[0]):
        utts.append(context[b_id, context_lens[b_id]-1])
    return np.array(utts)

def np2var(inputs, dtype, use_gpu):
    if inputs is None:
        return None
    return cast_type(Variable(th.from_numpy(inputs)), 
                     dtype, 
                     use_gpu)
