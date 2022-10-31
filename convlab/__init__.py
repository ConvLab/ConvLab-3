
import os
from convlab.nlu import NLU
from convlab.dst import DST
from convlab.policy import Policy
from convlab.nlg import NLG
from convlab.dialog_agent import Agent, PipelineAgent
from convlab.dialog_agent import Session, BiSession, DealornotSession

from os.path import abspath, dirname


def get_root_path():
    return dirname(dirname(abspath(__file__)))


DATA_ROOT = os.path.join(get_root_path(), 'data')
