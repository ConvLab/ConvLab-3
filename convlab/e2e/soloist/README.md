# SOLOIST

On top of the pre-trained LMs, SOLOIST subsumes different components of task-oriented dialogs into a single model and emplies a pre-training then fine-tuning schema to build task bots.

## Usage

Follow the instruction under each dataset's directory to prepare data training and evaluation.

#### Dataset Creation
Create datasets of three settings. 
```sh
$ cd multiwoz
$ python script/create_dataset.py joint
$ python script/create_dataset.py transfer
$ python script/create_dataset.py single
```

#### Train a model

```sh
$ python train.py --model_name_or_path t5-base --dataset_name e2e_dataloader.py --output_dir ./model --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --max_target_length 128 --max_length 512 --num_train_epochs 50 --save_steps 10000 --preprocessing_num_workers 1 --num_beams 5 --learning_rate 5e-5 --dataset_config_name SINGLE --logging_steps 100
```

The model (`pytorch_model.bin`) will be saved under the `output_dir` of the config file. The script will save predictions for validation/test every epoch.

#### Test a model

The result will be saved under the `output_dir` of the config file. For evaluation, a 3rd party package is used. Please follow the instructions at https://github.com/Tomiinek/MultiWOZ_Evaluation


## Performance on unified format datasets of different settings

 Note that we use almost the same hyper-parameters for different settings, which may not be optimal.

<table>
<thead>
  <tr>
    <th></th>
    <th colspan=2>MultiWOZ 2.1</th>
    <th colspan=2>SGD</th>
    <th colspan=2>Taskmaster-1</th>
  </tr>
</thead>
<thead>
  <tr>
    <th>Model</th>
    <th>Combined</th><th>BLEU</th>
    <th>Slot F1</th><th>BLEU</th>
    <th>Slot F1</th><th>BLEU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SOLOIST w/o pre-training</td>
    <td>67.0</td><td>16.8</td>
    <td>56.9</td><td>11.2</td>
    <td>8.5</td><td>28.0</td>    
  </tr>
  <tr>
    <td>SOLOIST </td>
    <td>71.4</td><td>17.1</td>
    <td>69.7</td><td>23.1</td>
    <td>9.2</td><td>29.2</td>

  </tr>
</tbody>
</table>

- Slot F1: F1 measure of the delexicalized slot predictions over the corpus.

## References

```
@article{peng2021soloist,
  title={Soloist: Buildingtask bots at scale with transfer learning and machine teaching},
  author={Peng, Baolin and Li, Chunyuan and Li, Jinchao and Shayandeh, Shahin and Liden, Lars and Gao, Jianfeng},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={807--824},
  year={2021},
  publisher={MIT Press}
}
@article{nekvinda2021shades,
  title={Shades of BLEU, flavours of success: The case of MultiWOZ},
  author={Nekvinda, Tom{\'a}{\v{s}} and Du{\v{s}}ek, Ond{\v{r}}ej},
  journal={arXiv preprint arXiv:2106.05555},
  year={2021}
}
@article{peng2022godel,
  title={GODEL: Large-Scale Pre-Training for Goal-Directed Dialog},
  author={Peng, Baolin and Galley, Michel and He, Pengcheng and Brockett, Chris and Liden, Lars and Nouri, Elnaz and Yu, Zhou and Dolan, Bill and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2206.11309},
  year={2022}
}
```
