# LLMforUS
Training large language model for user simulators.

## Train
We use `accelerate` and `peft` for training large language models.

You should use `accelerate config` to setup training configs, then `accelerate launch accel_train_model.py --model-args` to train the model.

More information is given in `train.sh`

## Evaluation
We should load two model path, `model-checkpoint` for LLM and `peft-checkpoint` for the adaptor.

Generate the results:
```
python3 generate_result.py --data $TEST_FILE --model-checkpoint $LLM_PATH --peft-checkpoint $ADAPTOR_PATH --result-dir $RESULT_DIR
```
The default result folder is `$ADAPTOR_PATH/result`


Evaluate the results:
```
python3 evaluation.py --data $RESULT_FOR_EVALUATION --evaluation-type $TYPE --result-dir RESULT_DIR
```
`$RESULT_FOR_EVALUATION` is generated from the previous stage (`Generate the results`).
`$TYPE` can be `emotion` or `nlg` (will support semantic actions).
The default folder of `RESULT_DIR` is the folder of `$RESULT_FOR_EVALUATION`
