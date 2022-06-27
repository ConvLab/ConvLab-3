## LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization
Codebase for [LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization](https://), published as a long paper in COLING 2020. The code is developed based on the implementations of the [LaRL](https://arxiv.org/abs/1902.08858) paper.

### Requirements
    python 3
    pytorch
    numpy
            
### Data
The pre-processed MultiWoz 2.0 data is included in data.zip. Unzip the compressed file and access the data under **data/norm-multi-woz**.
            
### Over structure:
The implementation of the models, as well as training and evaluation scripts are under **latent_dialog**.
The scripts for running the experiments are under **experiment_woz**. The trained models and evaluation results are under **experiment_woz/sys_config_log_model**.

There are 3 types of training to achieve the final model.

### Step 1: Unsupervised training (variational auto-encoding (VAE) task)
Given a dialogue response, the model is tasked to reproduce it via latent variables. With this task we aim to unsupervisedly capture generative factors of dialogue responses.

    - sl_cat_ae.py: train a VAE model using categorical latent variable
    - sl_gauss_ae.py: train a VAE model using continuous (Gaussian) latent variable

### Step 2: Supervised training (response generation task)
The supervised training step of the variational encoder-decoder model could be done 4 different ways. 

1. from scratch:


    - sl_word: train a standard encoder decoder model using supervised learning (SL)
    - sl_cat: train a latent action model with categorical latent variables using SL,
    - sl_gauss: train a latent action model with continuous latent varaibles using SL,

2. using the VAE models as pre-trained model (equivalent to LAVA_pt):


    - finetune_cat_ae: use the VAE with categorical latent variables as weight initialization, and then fine-tune the model on response generation task
    - finetune_gauss_ae: as above but with continuous latent variables 
    - Note: Fine-tuning can be set to be selective (only fine-tune encoder) or not (fine-tune the entire network) using the "selective_finetune" argument in config

3. using the distribution of the VAE models to obtain informed prior (equivalent to LAVA_kl):


    - actz_cat: initialized new encoder is combined with pre-trained VAE decoder and fine-tuned on response generation task. VAE encoder is used to obtain an informed prior of the target response and not trained further.
    - actz_gauss: as above but with continuous latent variables

4. or simultaneously from scrath with VAE task in a multi-task fashion (equivalent to LAVA_mt):


    - mt_cat: train a model to optimize both auto-encoding and response generation in a multi-task fashion, using categorical latent variables
    - mt_gauss: as above but with continuous latent variables

No.1 and 4 can be directly trained without Step 1. No. 2 and 3 requires a pre-trained VAE model, given via a dictionary 

    pretrained = {"2020-02-26-18-11-37-sl_cat_ae":100}

### Step 3: Reinforcement Learning
The model can be further optimized with RL to maximize the dialogue success.

Each script is used for:

    - reinforce_word: fine tune a pretrained model with word-level policy gradient (PG)
    - reinforce_cat: fine tune a pretrained categorical latent action model with latent-level PG.
    - reinforce_gauss: fine tune a pretrained gaussian latent action model with latent-level PG.

The script takes a file containing list of test results from the SL step.

    f_in = "sys_config_log_model/test_files.lst"


### Checking the result
The evaluation result can be found at the bottom of the test_file.txt. We provide the best model in this repo.

NOTE: when re-running the experiments some variance is to be expected in the numbers due to factors such as random seed and hardware specificiations. Some methods are more sensitive to this than others.
