# -*- coding: utf-8 -*-
import zipfile
import logging
import torch
import os
import json

from convlab.policy.policy import Policy
from convlab.util.file_util import cached_path
from convlab.policy.rlmodule import MultiDiscretePolicy
from convlab.policy.vector.vector_binary import VectorBinary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "mle_policy_multiwoz.zip")


class MLEAbstract(Policy):

    def __init__(self, vector, policy):
        self.vector = vector
        self.policy = policy

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec, m = self.vector.state_vectorize(state)
        s_vec = torch.Tensor(s_vec)
        m = torch.from_numpy(m).to(DEVICE)
        a = self.policy.select_action(s_vec.to(device=DEVICE), False, action_mask=m).cpu()
        action = self.vector.action_devectorize(a.detach().numpy())
        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MLE Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(os.path.join(model_dir, 'best_mle.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))

    def load(self, filename):
        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_mle.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break


class MLE(MLEAbstract):

    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)

        self.vector = VectorBinary()
        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)

    @classmethod
    def from_pretrained(cls,
                        archive_file=DEFAULT_ARCHIVE_FILE,
                        model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/mle_policy_multiwoz.zip'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        model = cls()
        model.load_from_pretrained(archive_file, model_file, cfg['load'])
        return model


class MLEPolicy(MLE):
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/mle_policy_multiwoz.zip'):
        super().__init__()
        if model_file:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
                cfg = json.load(f)
            self.load_from_pretrained(archive_file, model_file, cfg['load'])
