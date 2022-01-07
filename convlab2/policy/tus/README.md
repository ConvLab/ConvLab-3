**TUS** is a domain-independent user simulator with transformers for task-oriented dialogue systems. It is based on the [ConvLab-2](https://github.com/thu-coai/ConvLab-2) framework. Therefore, you should follow their instruction to install the package.

## Introduction
Our model is a domain-independent user simulator, which means it is not based on any domain-dependent freatures and the output representation is also domain-independent. Therefore, it can easily adapt to a new domain, without additional feature engineering and model retraining.

The code of TUS is in `convlab2/policy/tus` and a rule-based DST of user is also created in `convlab2/dst/rule/multiwoz/dst.py` based on the rule-based DST in `convlab2/dst/rule/multiwoz/dst.py`.

## How to run the model
### Train the user simulator
`python3 convlab2/policy/tus/multiwoz/train.py --user_config convlab2/policy/tus/multiwoz/exp/default.json`

One default configuration is placed in `convlab2/policy/tus/multiwoz/exp/default.json`. They can be modified based on your requirements. For example, the output directory can be specified in the configuration (`model_dir`).

### Train a dialogue policy with TUS
You can use it as a normal user simulator by `PipelineAgent`. For example,
```python
import json
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dst.rule.multiwoz.usr_dst import UserRuleDST
from convlab2.policy.tus.multiwoz.TUS import UserPolicy

user_config_file = "convlab2/policy/tus/multiwoz/exp/default.json"
dst_usr = UserRuleDST()
user_config = json.load(open(user_config_file))
policy_usr = UserPolicy(user_config)
simulator = PipelineAgent(None, dst_usr, policy_usr, None, 'user')
```


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
