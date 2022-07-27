# Our paper
[Uncertainty Measures in Neural Belief Tracking and the Effects on Dialogue Policy Performance](https://todo.pdf)

## Structure
![SetSUMBT Architecture](https://gitlab.cs.uni-duesseldorf.de/dsml/convlab-2/-/raw/develop/convlab/dst/setsumbt/setSUMBT.png?inline=false)

## Usages
### Data preprocessing
We conduct experiments on the following datasets:

* MultiWOZ 2.1 [Download](https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip) to get `MULTIWOZ2.1.zip`

### Train
**Train baseline single instance SetSUMBT**
```
python run.py --run_nbt \
    --use_descriptions --set_similarity \
    --do_train --do_eval \
    --seed 20211202
```

**Train ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
./configure_ensemble.sh $SEED $MODEL_PATH
./train_ensemble.sh $SEED $MODEL_PATH
```

**Distill Ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
./distill_end.sh $SEED $MODEL_PATH
```

**Distribution Distill Ensemble SetSUMBT**
```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
./distill_end2.sh $SEED $MODEL_PATH
```

### Evaluation

```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
python run.py --run_calibration \
    --seed $SEED \
    --output_dir $MODEL_PATH
```

### Convert training setup to convlab model

```
SEED=20211202
MODEL_PATH="models/SetSUMBT-CE-roberta-gru-cosine-labelsmoothing-Seed$SEED-$(date +'%d-%m-%Y')"
OUT_PATH="models/labelsmoothing"
./configure_model.sh $MODEL_PATH data $OUT_PATH
```

### Training PPO policy using SetSUMBT tracker and uncertainty

To train a PPO policy switch to the directory:
```
cd ../../policy/ppo
```
In this directory run the relevant train script, for example to train the policy using END-SetSUMBT using no uncertainty metrics run:
```
./train_setsumbt_end_baseline.sh
```
