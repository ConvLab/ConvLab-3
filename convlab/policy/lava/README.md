## LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization
ConvLab3 interface for [LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization](https://aclanthology.org/2020.coling-main.41/), published as a long paper in COLING 2020.

To train a LAVA model, clone and follow instructions from the [original LAVA repository](https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public).

With a (pre-)trained LAVA model, it is possible to evaluate or perform online RL with ConvLab3 US by loading the lava module with

- from convlab.policy.lava.multiwoz import LAVA

and using it as the policy module in the ConvLab pipeline (NLG should be set to None).


Code example can be found at
- ConvLab-3/examples/agent_examples/test_LAVA.py

A trained LAVA model can be found at https://huggingface.co/ConvLab/lava-policy-multiwoz21.

