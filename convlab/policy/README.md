# Dialog Policy

In the pipeline task-oriented dialog framework, the dialogue policy module
takes as input the dialog state, and chooses the system action based on
it. This directory contains the interface definition of dialogue policy
module for both system side and user simulator side, as well as some
implementations under different sub-directories. 

An important additional module for the policy is the vectoriser which translates the dialogue state into a vectorised form that the dialogue policy network expects as input. 
Moreover, it translates the vectorised act that the policy took back into semantic form. More information can be found in the directory /convlab/policy/vector.

We currently maintain the following policies:

**system policies**: GDPL, MLE, PG, PPO and VTRACE DPT

**user policies**: rule, TUS, GenTUS


## Overview

Every policy directory typically has two python scripts, **1) train.py** for running an RL training and **2) a dedicated script** that implements the algorithm and loads the policy network (e.g. **ppo.py** for the ppo policy).
Moreover, two config files define the environment and the hyper parameters for the algorithm:

- config.json: defines the hyper parameters such as learning rate and algorithm related parameters
- environment.json: defines the learning environment (MDP) for the policy. This includes the NLU, DST and NLG component for both system and user policy as well as which user policy should be used. It also defines the number of total training dialogues as well as evaluation dialogues and frequency.

An example for the environment.json is the **semantic_level_config.json** in the policy subfolders.



## Workflow

The workflow can be generally decomposed into three steps that will be explained in more detail below:

1. set up the environment configuration and policy parameters
2. run a reinforcement learning training with the given configurations
3. evaluate your trained models

#### Set up the environment 

The necessary step before starting a training is to set up the environment and policy parameters. Information about policy parameters can be found in each policy subfolder. The following example defines an environment for the policy with the rule-based dialogue state tracker, no NLU, no NLG, and the rule-based user simulator:

```
{
	"model": {
		"load_path": "", # specify a loading path to load a pre-trained model, omit the ending .pol.mdl
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

Once you set up your configuration, you are ready to start an experiment by executing

```sh
$ python convlab/policy/policy_subfolder/train.py --path=your_environment_config --seed=your_seed
```

You can specify the seed either in the environment config or through the argument parser. If you do not specify an environment config, it will automatically load the default config. 

Once the training started, it will automatically generate an **experiment** folder and a corresponding experiment-TIMESTEP folder in it. Inside of that, there are 4 subfolders configs, logs, save and TB_summary:

- **configs**: containts information about which config was used
- **logs**: will save information created by a logger during training
- **save**: a folder for saving model checkpoints
- **TB_summary**: saves a tensorboard summary that will be later used for plotting graphs

Once the training finished, it will move the experiment-TIMESTAMP folder into the **finished_experiments** folder.

#### Evaluating your models

The evaluation tools can be found in the folder convlab/policy/plot_results. Please have a look in the README for detailed instructions. 

#### Running Evaluation Dialogues

You can run evaluation dialogues with a trained model using 

```sh
$ python convlab/policy/evaluate.py --model_name=NAME --config_path=PATH --num_dialogues=NUM --verbose
```

- model_name: specify which model is used, i.e. MLE, PPO, PG, DDPT
- config_path: specify the config-path that was used during RL training, for instance semantic_level_config.json
- num_dialogues: number of evaluation dialogues
- verbose: can be also excluded. If used, it will print the dialogues in the termain consoloe together with its goal. That helps in analysing the behaviour of the policy.

## Adding a new policy

If you would like to add a new policy, start by creating a subfolder for it. Then make sure that you have the four files mentioned in **Overview** section in it.

#### Algorithm script

Here you define your algorithm and policy network. Please ensure that you also load a vectoriser here that is inherited from the vector/vector_base.py class. 

In addition, your policy module is required to have a **predict** method where the skeleton usually looks something like:

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        
        # uses the vector class for vectorisation of the dialogue state and also creates an action mask
        s, action_mask = self.vector.state_vectorize(state) 
        s_vec = torch.Tensor(s)
        mask_vec = torch.Tensor(action_mask)
        
        # predict an action using the policy network
        a = self.policy.select_action(s_vec, mask_vec)

        # map the action indices back to semantic actions using the vectoriser
        action = self.vector.action_devectorize(a.detach().numpy())
        return action

#### train.py script

The train.py script is responsible for several different functions. In the following we will provide some code or pointers on how to do these steps. Have a look at the train.py files as well.

1. load the config and set seed
    ```
    environment_config = load_config_file(path)
    conf = get_config(path, args)
    seed = conf['model']['seed']
    set_seed(seed)
   save_config(vars(parser.parse_args()), environment_config, config_save_path)
    ```

2. saves additional information (through a logger and tensorboard writer)

    ```
    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)
   ```   
   
   
3. load the policy module

    ```
    policy_sys = PPO(True, seed=conf['model']['seed'], vectorizer=conf['vectorizer_sys_activated'])
    ```
4. load the environment using th environment-config
    ```
   env, sess = env_config(conf, policy_sys)
   ```

5. collect dialogues and execute policy updates: use the update function of policy_sys and implement a create_episodes function.
6. run evaluation during training and save policy checkpoints

    ```
    logging.info(f"Evaluating after Dialogues: {num_dialogues} - {time_now}" + '-' * 60)
    eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
    best_complete_rate, best_success_rate, best_return = \
        save_best(policy_sys, best_complete_rate, best_success_rate, best_return,
                  eval_dict["complete_rate"], eval_dict["success_rate_strict"],
                  eval_dict["avg_return"], save_path)
    policy_sys.save(save_path, "last")
    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], idx * conf['model']['batchsz'])
    ```

