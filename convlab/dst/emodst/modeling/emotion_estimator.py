import torch
import numpy as np
import os

from convlab.dst.emodst.modeling.erc_models import ContextBERT_ERToD

ERC_MODELS = {
    'contextbert-ertod': ContextBERT_ERToD
}


class EmotionEstimator():
    def __init__(self,
                 kwargs_for_model: dict = {}) -> None:

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = ERC_MODELS[kwargs_for_model['model_type']](
            **kwargs_for_model).to(self.device)
        if os.path.exists(kwargs_for_model['model_name_or_path']):
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(
                    kwargs_for_model['model_name_or_path'])['state_dict'])
            else:
                self.model.load_state_dict(torch.load(kwargs_for_model['model_name_or_path'],
                                                      map_location=torch.device('cpu'))['state_dict'])
        else:
            raise NameError('ERCModelCheckpointNotFound')
        # self.model.save_config('/home/shutong/models/contextbert-ertod/config.json')
        # self.model.save_pretrained('/home/shutong/models/contextbert-ertod')
        self.model.eval()

    def predict(self,
                user_utt: str = "",
                dialog_state_history: list = []) -> str:

        input_ids, attn_mask, ds_vec = self.model.normalise_input(
            user_utt, dialog_state_history
        )
        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)
        ds_vec = ds_vec.to(self.device)

        emo_logits, _, _, _ = self.model(
            input_ids=input_ids, attention_mask=attn_mask, ds=ds_vec)
        _, emo_pred = torch.max(emo_logits, dim=1)

        return emo_pred
