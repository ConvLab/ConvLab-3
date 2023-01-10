## EmoWOZ

This is the codebase for [EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems](https://arxiv.org/abs/2109.04919). 


### Data

The dataset can be found in `data/`. EmoWOZ adopts the same format as MultiWOZ logs. We add an additional `emotion` field in each log item. The emotion contains annotations by three annotators, each identified by an anonymous 8-character global annotator id. The `final` field contains the final label obtained either from majority voting or manual resolution.

All DialMAGE dialogues have a dialogue id in the form of ''DMAGExxx.json'' where xxx is a number. We provide dialog_act and span_info used to generate system responses in DialMAGE. 

The definition for each label is defined as below:
| Label | Emotion Tokens               | Valence  | Elicitor   | Conduct  |
|-------|------------------------------|----------|------------|----------|
| 0     | Neutral                      | Neutral  | Any        | Polite   |
| 1     | Fearful, sad, disappointed   | Negative | Event/fact | Polite   |
| 2     | Dissatisfied, disliking      | Negative | Operator   | Polite   |
| 3     | Apologetic                   | Negative | User       | Polite   |
| 4     | Abusive                      | Negative | Operator   | Impolite |
| 5     | Excited, happy, anticipating | Positive | Event/fact | Polite   |
| 6     | Satisfied, liking            | Positive | Operator   | Polite   |

EmoWOZ dataset is licensed under Creative Commons Attribution-NonCommercial 4.0 International Public License and later.


### Baseline Models

To test the dataset with baseline models used in the paper, please follow instructions in each model folder of `baselines/`.
The implementation of two models, `baselines/COSMIC/` and `baselines/DialogueRNN/`, are taken and modified from https://github.com/declare-lab/conv-emotion.

### Requirements

See `requirements.txt`. These are packages required for running all baseline models. Tested versions are listed below:
- Python (tested: 3.7.8)
- transformers (tested: 4.12.5)
- torch (tested: 1.8.1)
- pandas (tested: 1.3.4)
- sklearn (tested: 1.0.1)
- tqdm (tested: 4.62.3)
- nltk (tested: 3.6.5)
- ftfy (tested: 6.0.3)
- spacy (tested: 3.2.0)
- ipython (tested: 7.30.1)
- keras (tested: 2.7.0)
- tensorflow (2.7.0)


### Citation

If you use EmoWOZ in your own work, please cite our work as follows:

```
@misc{feng2021emowoz,
      title={EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion in Task-Oriented Dialogue Systems}, 
      author={Shutong Feng and Nurul Lubis and Christian Geishauser and Hsien-chin Lin and Michael Heck and Carel van Niekerk and Milica Gašić},
      year={2021},
      eprint={2109.04919},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Please note that this dataset should only be used for research purpose.


### Contact

Any questions or bug reports can be sent to shutong.feng@hhu.de