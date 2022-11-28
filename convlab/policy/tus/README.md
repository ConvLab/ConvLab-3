**TUS** is a domain-independent user simulator with transformers for task-oriented dialogue systems.

## Introduction
Our model is a domain-independent user simulator, which means its input and output representations are domain agnostic. Therefore, it can easily adapt to a new domain, without additional feature engineering and model retraining.

The code of TUS is in `convlab/policy/tus`.

## Usage
### Train TUS from scratch

```
python3 convlab/policy/tus/unify/train.py --dataset $dataset --dial-ids-order $dial_ids_order --split2ratio $split2ratio --user-config $config
```

`dataset` can be `multiwoz21`, `sgd`, `tm`, `sgd+tm`, or `all`.
`dial_ids_order` can be 0, 1 or 2
`split2ratio` can be 0.01, 0.1 or 1
Default configurations are placed in `convlab/policy/tus/unify/exp`. They can be modified based on your requirements. 

For example, you can train TUS for multiwoz21 by 
`python3 convlab/policy/tus/unify/train.py --dataset multiwoz21 --dial-ids-order 0 --split2ratio 1 --user-config "convlab/policy/tus/unify/exp/multiwoz.json"`

### Evaluate TUS

### Train a dialogue policy with TUS
You can use it as a normal user simulator by `PipelineAgent`. For example,
```python
import json
from convlab.dialog_agent.agent import PipelineAgent
from convlab.policy.tus.unify.TUS import UserPolicy

user_config_file = "convlab/policy/tus/unify/exp/multiwoz.json"
user_config = json.load(open(user_config_file))
policy_usr = UserPolicy(user_config)
simulator = PipelineAgent(None, None, policy_usr, None, 'user')
```
then you can train your system with this simulator.

There is an example config, which trains a PPO policy with TUS in semantic level, in `convlab/policy/ppo/tus_semantic_level_config.json`.
You can train a PPO policy as following, 
```
config="convlab/policy/ppo/tus_semantic_level_config.json"
python3 convlab/policy/ppo/train.py --path $config
```
notice: You should name your pretrained policy as `convlab/policy/ppo/pretrained_models/mle` or modify the `load_path` of `model` in the config `convlab/policy/ppo/tus_semantic_level_config.json`.


<!---citation--->
## Citing

```
@inproceedings{lin-etal-2021-domain,
    title = "Domain-independent User Simulation with Transformers for Task-oriented Dialogue Systems",
    author = "Lin, Hsien-chin and Lubis, Nurul and Hu, Songbo and van Niekerk, Carel and Geishauser, Christian and Heck, Michael and Feng, Shutong and Gasic, Milica",
    booktitle = "Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    year = "2021",
    url = "https://aclanthology.org/2021.sigdial-1.47",
    pages = "445--456"
}

```

## License

Apache License 2.0
