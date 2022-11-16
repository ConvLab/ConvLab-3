# Dialog Policy

In the pipeline task-oriented dialog framework, the dialogue policy module
takes as input the dialog state, and chooses the system action based on
it. 

The so-called vectoriser translates the dialogue state into a vectorised form that the dialogue policy network expects as input. 
Moreover, it translates the vectorised act that the policy took back into semantic form. More information can be found in the directory /convlab/policy/vector.

This directory contains the interface definition of dialogue policy
module for both system side and user simulator side, as well as some
implementations under different sub-directories. 

We currently maintain the following policies:
system policies: GDPL, MLE, PG, PPO and VTRACE DPT

user policies: rule, TUS, GenTUS


## Interface

The interfaces for dialog policy are defined in policy.Policy:

- **predict** takes as input agent state (often the state tracked by DST)
and outputs the next system action.

Every policy directory typically has two python scripts, **1) train.py** for running an RL training and **2) a dedicated script** that implements the algorithm and loads the policy network (e.g. **ppo.py** for the ppo policy).
Moreover, two config files define the environment and the hyper parameters for the algorithm:

- config.json: defines the hyper parameters such as learning rate and algorithm related parameters
- environment.json: defines the learning environment (MDP) for the policy. This includes the NLU, DST and NLG component for both system and user policy as well as which user policy should be used. It also defines the number of total training dialogues as well as evaluation dialogues and frequency.

An example for the environment.json is the **semantic_level_config.json** in the policy subfolders.



## Workflow

The workflow can be generally decomposed into three steps that will be explained in more detail below:

- set up the environment configuration and policy parameters
- run a reinforcement learning training with the given configurations
- evaluate your trained models

#### Set up the environment 

The necessary step before starting a training is to set up the environment and policy parameters. Information about policy parameters can be found in each policy subfolder. The following example defines an environment for the policy with the rule-based dialogue state tracker, no NLU, no NLG, and the rule-based user simulator:

```
{
	"model": {
		"load_path": "", # specify a loading path to load a pre-trained model 
		"use_pretrained_initialisation": false, # will download a provided ConvLab-3 model
		"pretrained_load_path": "",
		"seed": 0, # the seed for the experiment
		"eval_frequency": 5, # how often evaluation should take place
		"process_num": 4, # how many processes the evaluation should use for speed up
		"sys_semantic_to_usr": false,
		"num_eval_dialogues": 500 # how many dialogues should be used for evaluation
	},
	"vectorizer_sys": {
		"uncertainty_vector_mul": {
			"class_path": "convlab.policy.vector.vector_binary.VectorBinary",
			"ini_params": {
				"use_masking": true,
				"manually_add_entity_names": false,
				"seed": 0
			}
		}
	},
	"nlu_sys": {},
	"dst_sys": {
		"RuleDST": {
			"class_path": "convlab.dst.rule.multiwoz.dst.RuleDST",
			"ini_params": {}
		}
	},
	"sys_nlg": {},
	"nlu_usr": {},
	"dst_usr": {},
	"policy_usr": {
		"RulePolicy": {
			"class_path": "convlab.policy.rule.multiwoz.RulePolicy",
			"ini_params": {
				"character": "usr"
			}
		}
	},
	"usr_nlg": {}
}
```

#### Executing a training

Once you set up your configuration, you are good to start an experiment by executing

```python convlab/policy/policy_subfolder/train.py --path=your_environment_config --seed=your_seed```

You can specify the seed either in the environment config or through the argument parser. If you do not specify an environment config, it will automatically load the default config.