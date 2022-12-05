# SetSUMBT & SUMBT
## Dialogue State Tracking and Language Understanding

The SUMBT and SetSUMBT models is a group of dialogue state tracking models.
These models include natural language understanding prediction heads which
provide crucial information, such as the user request actions, required to
incorporate the model in a pipeline dialogue system. [SUMBT](https://arxiv.org/pdf/1907.07421.pdf)
utilises a Slot-Utterance matching attention mechanism (SUM) for information extraction,
a recurrent module for latent information tracking and a picklist state
prediction head using similarity based matching. [SetSUMBT](https://aclanthology.org/2021.emnlp-main.623/)
extends the SUMBT model through the extension of the Slot-Utterance matching
using to a set based Slot-Utterance matching module and a set based similarity
matching prediction head. This model also introduces the language understanding
prediction heads required for predicting additional crucial information. In addition,
this model code allows for training of an ensemble and distillation of the ensemble
producing a belief tracking model which predicts well calibrated belief states.


## Our paper
[Uncertainty Measures in Neural Belief Tracking and the Effects on Dialogue Policy Performance](https://aclanthology.org/2021.emnlp-main.623/)

## SetSUMBT Model Architecture
![SetSUMBT Architecture](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/setSUMBT.png?raw=true)
The default configuration of the models are as follows:

| Hyperparameter              |                      SetSUMBT                       |                             SUMBT                             |
|:----------------------------|:---------------------------------------------------:|:-------------------------------------------------------------:|
| Max Turns                   |                         12                          |                              12                               |
| Max Turn Length             |                         64                          |                              64                               |
| Max Candidate Desc. Length  |                         12                          |                              12                               |
| Encoder model               | [roberta-base](https://huggingface.co/roberta-base) | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| Hidden Size                 |                         768                         |                              768                              |
| SUM Attention Heads         |                         12                          |                              12                               |
| Dropout rate                |                         0.3                         |                              0.3                              |
| Tracker type                |                         GRU                         |                              GRU                              |
| Tracker Hidden Size         |                         300                         |                              300                              |
| Tracker RNN Layers          |                          1                          |                               1                               |
| Set Pooler type             |                         CNN                         |                         No Set Pooler                         |
| Candidate Desc. Pooler type |                      No Pooler                      |                           CLS Token                           |
| Loss Function               |                   Label smoothing                   |                        Label smoothing                        |
| Epochs                      |                         50                          |                              50                               |
| Early stopping criteria     |                      20 Epochs                      |                           20 Epochs                           |
| Learning rate               |                        5e-5                         |                             5e-5                              |
| LR Scheduler                |                     Linear(0.2)                     |                          Linear(0.2)                          |

## Usages
### Data sets
We conduct experiments on the following datasets:

* [MultiWOZ 2.1](https://huggingface.co/datasets/ConvLab/multiwoz21)
* [Schema Guided Dialogue(SGD)](https://huggingface.co/datasets/ConvLab/sgd)
* [Taskmaster 1](https://huggingface.co/datasets/ConvLab/tm1)
* [Taskmaster 2](https://huggingface.co/datasets/ConvLab/tm2)
* [Taskmaster 3](https://huggingface.co/datasets/ConvLab/tm3)

### Model checkpoints available on Huggingface

The following pre-trained model checkpoints are available via huggingface hub:

| Model    | Dataset      | Training Setup                   | Checkpoint                                                                        |
|:---------|:-------------|:---------------------------------|:----------------------------------------------------------------------------------|
| SetSUMBT | MultiWOZ 2.1 | Full dataset                     | [setsumbt-dst-multiwoz21](https://huggingface.co/ConvLab/setsumbt-dst-multiwoz21) |
| SetSUMBT | SGD          | Full dataset                     | [setsumbt-dst-sgd](https://huggingface.co/ConvLab/setsumbt-dst-sgd)               |
| SetSUMBT | TM1+TM2+TM3  | Full dataset                     | [setsumbt-dst-tm123](https://huggingface.co/ConvLab/setsumbt-dst-tm123)           |
| SetSUMBT | MultiWOZ 2.1 | DST+NLU tasks + Uncertainty Est. | [setsumbt-dst_nlu-multiwoz21-EnD2](https://huggingface.co/ConvLab/setsumbt-dst_nlu-multiwoz21-EnD2)           |

### Train
**Train baseline single instance SetSUMBT**

Command to train the model on the MultiWOZ 2.1 dataset, to train the model on
other datasets/setups or to train the SUMBT model set the relevant `starting_config_name`.
To fine tune a pre-trained model set the `model_name_or_path` to the path of the pre-trained
model. See below for more configurations of this model:

| Model    | Dataset              | Training Setup                     | Starting Config Name                                                                                                                            |
|:---------|:---------------------|:-----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| SetSUMBT | MultiWOZ21           | Full dataset                       | [setsumbt_multiwoz21](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask.json)                    |
| SetSUMBT | MultiWOZ21           | DST and NLU Tasks                  | [setsumbt_nlu_multiwoz21](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_nlu_multiwoz21.json)           |
| SetSUMBT | MultiWOZ21           | Ensemble Distillation              | [setsumbt_nlu_multiwoz21_end](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_nlu_multiwoz21_end.json)   |
| SetSUMBT | MultiWOZ21           | Ensemble Distribution Distillation | [setsumbt_nlu_multiwoz21_end2](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_nlu_multiwoz21_end2.json) |
| SetSUMBT | MultiWOZ21           | 10% of the training data           | [setsumbt_multiwoz21_10p](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask_10p.json)            |
| SetSUMBT | MultiWOZ21           | 1% of the training data            | [setsumbt_multiwoz21_1p](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask_1p.json)              |
| SetSUMBT | TM1+TM2+TM3          | Full dataset                       | [setsumbt_tm](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask.json)                            |
| SetSUMBT | SGD                  | Full dataset                       | [setsumbt_sgd](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask.json)                           |
| SetSUMBT | MW21+SGD+TM1+TM2+TM3 | Joint training                     | [setsumbt_joint](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask.json)                         |
| SetSUMBT | SGD+TM1+TM2+TM3      | Pre training                       | [setsumbt_pretrain](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/setsumbt_multitask.json)                      |
| SUMBT    | MultiWOZ21           | Full dataset                       | [sumbt_multiwoz21](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask.json)                          |
| SUMBT    | MultiWOZ21           | 10% of the training data           | [sumbt_multiwoz21_10p](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask_10p.json)                  |
| SUMBT    | MultiWOZ21           | 1% of the training data            | [sumbt_multiwoz21_1p](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask_1p.json)                    |
| SUMBT    | TM1+TM2+TM3          | Full dataset                       | [sumbt_tm](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask.json)                                  |
| SUMBT    | SGD                  | Full dataset                       | [sumbt_sgd](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask.json)                                 |
| SUMBT    | MW21+SGD+TM1+TM2+TM3 | Joint training                     | [sumbt_joint](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask.json)                               |
| SUMBT    | SGD+TM1+TM2+TM3      | Pre training                       | [sumbt_pretrain](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/dst/setsumbt/configs/sumbt_multitask.json)                            |

```
python3 run.py \
    --starting_config_name setsumbt_multiwoz21 \
    --seed 0 \
    --do_train
```

**Train ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
ENSEMBLE_SIZE=10
DATA_SIZE=7500

python3 run.py \
    --starting_config_name setsumbt_nlu_multiwoz21 \
    --output_dir $MODEL_PATH \
    --ensemble_size $ENSEMBLE_SIZE \
    --data_sampling_size $DATA_SIZE \
    --seed $SEED

ENSEMBLE_SIZE=$(($ENSEMBLE_SIZE-1))
for e in $(seq 0 $ENSEMBLE_SIZE);do
    python3 run.py \
        --starting_config_name setsumbt_nlu_multiwoz21
        --output_dir "$OUT/ens-$e" \
        --do_train \
        --seed $SEED
done
```

**Distill Ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
for SUBSET in train dev test;do
    python3 distillation_setup.py \
        --model_path $MODEL_PATH \
        --set_type $SUBSET \
        --reduction mean \
        --get_ensemble_distributions \
        --convert_distributions_to_predictions
done
python3 run.py \
    --starting_config_name setsumbt_nlu_multiwoz21_end \
    --seed $SEED \
    --output_dir $MODEL_PATH \
    --do_train
```

**Distribution Distill Ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
for SUBSET in train dev test;do
    python3 distillation_setup.py \
        --model_path $MODEL_PATH \
        --set_type $SUBSET \
        --reduction none \
        --get_ensemble_distributions \
        --convert_distributions_to_predictions
done
python3 run.py \
    --starting_config_name setsumbt_nlu_multiwoz21_end2 \
    --seed $SEED \
    --output_dir $MODEL_PATH \
    --do_train
```

### Evaluation

To evaluate a model set the `$MODEL_PATH` to the path or URL of that model.
The URL is the download URL of the model archive from the pretrained model
for example for `setsumbt-dst-multiwoz21` the url is
https://huggingface.co/ConvLab/setsumbt-dst-multiwoz21/resolve/main/SetSUMBT-multiwoz21-roberta-gru-cosine-labelsmoothing-Seed0.zip.

```
python3 run.py \
    --starting_config_name setsumbt_multiwoz21 \
    --output_dir $MODEL_PATH \
    --do_test
python3 get_golden_labels.py \
    --dataset_name multiwoz21 \
    --model_path $MODEL_PATH
python3 ../evaluate_unified_datasets.py \
    -p "$MODEL_PATH/predictions/test_multiwoz21.json"
```

### Training PPO policy using SetSUMBT tracker and uncertainty

To train a PPO policy switch to the directory:
```
cd ../../policy/ppo
```
In this directory run the relevant train script, for example to train the policy using END-SetSUMBT using no uncertainty metrics run:
```
python3 train.py --path setsumbt_config.json
```
