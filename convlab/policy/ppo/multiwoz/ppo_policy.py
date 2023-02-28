from convlab.policy.ppo import PPO
import os
import json

class PPOPolicy(PPO):
    def __init__(self,  is_train=False, 
                        dataset='Multiwoz', 
                        archive_file="",
                        model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/ppo_policy_multiwoz.zip"
                        ):
        super().__init__(is_train=is_train, dataset=dataset)
        self.load_from_pretrained(model_file)
        