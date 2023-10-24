import torch
from torch import nn

import numpy as np

from transformers import (BertModel, BertConfig, BertTokenizer,
                          RobertaModel, RobertaConfig, RobertaTokenizer)

from convlab.policy.vector.vector_binary import VectorBinary

MODEL_CLASSES = {
    'bert-base-uncased': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
    }

class BertERC(nn.Module):
    """ContextBERT-ERToD ERC Model"""

    def __init__(self,
                 model_name_or_path: str = "",
                 base_model_type: str = "bert-base-uncased",
                 use_context: bool = True,
                 use_dialog_state: bool = True,
                 mtl_valence: bool = True,
                 mtl_elicitor: bool = True,
                 mtl_conduct: bool = True,
                 max_token_len: int = 128,
                 ds_dim: int = 361,
                 ds_ctx_window_size: int = 3,
                 ds_output_dim: int = 256,
                 dropout_rate: float = 0.3):
        super(BertERC, self).__init__()
        
        self.use_dialog_state = use_dialog_state
        self.max_token_len = max_token_len
        self.ds_dim = ds_dim
        self.ds_ctx_window_size = ds_ctx_window_size
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[base_model_type]

        self.config = self.config_class.from_pretrained(base_model_type)
        self.bert = self.model_class.from_pretrained(base_model_type, config=self.config)
        
        if use_dialog_state:
            self.ds_projection = nn.Linear(ds_dim*ds_ctx_window_size, ds_output_dim)
            self.tanh = nn.Tanh()
            feature_dim = self.bert.config.hidden_size + ds_output_dim
        else:
            feature_dim = self.bert.config.hidden_size

        n_classes = 7
        self.drop = nn.Dropout(p=dropout_rate)   # define dropout
        self.out = nn.Linear(feature_dim, n_classes)   # linear layer for emotion classification
        self.out_valence = nn.Linear(feature_dim, 3)   # linear layer for valence classification
        self.out_elicitor = nn.Linear(feature_dim, 3)   # linear layer for elicitor classification
        self.out_conduct = nn.Linear(feature_dim, 2)   # linear layer for conduct classification

        self.tokenizer = self.tokenizer_class.from_pretrained(base_model_type)
        self.vectoriser = VectorBinary()

    def forward(self, input_ids, attention_mask, ds=None):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        if self.use_dialog_state:
            cls_feature = self.drop(pooled_output)
            ds_feature = self.tanh(self.ds_projection(ds))
            output = torch.cat((cls_feature, ds_feature), dim=1)
        else:
            output = self.drop(pooled_output)

        valence_logits = self.out_valence(output)
        elicitor_logits = self.out_elicitor(output)
        conduct_logits = self.out_conduct(output)
        emotion_logits = self.out(output)

        return emotion_logits, valence_logits, elicitor_logits, conduct_logits
    
    def normalise_input(self, 
                user_utt: str = "",
                dialog_state_history: list = []) -> {}:

        history = [utt for role, utt in dialog_state_history[-1]['history']]
        
        history_str = ""
        for i in reversed(range(len(history))):  # reverse order to place the current turn closer to the [CLS]
            if i%2 == 0:
                history_str += f"user: {history[i]} "   
            else:
                history_str += f"system: {history[i]} "
        text = f"user: {user_utt} {history_str}"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # todo: might need to use [0] for vectorised state
        dialog_states = [self.vectoriser.state_vectorize(state)[0] for state in dialog_state_history]
        
        ctx_ds_len = self.ds_ctx_window_size * self.use_dialog_state
        padded_dialog_states = [np.zeros((self.ds_dim,))] * (ctx_ds_len - len(dialog_states)) + dialog_states
        ctx_ds_vec = np.concatenate(tuple(padded_dialog_states[::-1][:ctx_ds_len]), axis=None)

        if len(ctx_ds_vec) == 0:
            ctx_ds_vec = None

        input_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        dialog_state_vec = torch.FloatTensor(ctx_ds_vec).unsqueeze(0)

        return input_ids, attention_mask, dialog_state_vec
