# ConvLab-3

![PyPI](https://img.shields.io/pypi/v/convlab)![PyPI - Python Version](https://img.shields.io/pypi/pyversions/convlab)![GitHub](https://img.shields.io/github/license/ConvLab/ConvLab-3)

**ConvLab-3** is a flexible dialog system platform based on a **unified data format** for task-oriented dialog (TOD) datasets. The unified format serves as the adapter between TOD datasets and models: datasets are first transformed to the unified format and then loaded by models. In this way, the cost of adapting $M$ models to $N$ datasets is reduced from $M\times N$ to $M+N$. While retaining all features of [ConvLab-2](https://github.com/thu-coai/ConvLab-2),  ConvLab-3 greatly enlarges supported datasets and models thanks to the unified format, and enhances the utility of reinforcement learning (RL) toolkit for dialog policy module. For typical usage, see our [paper](). Datasets and Trained models are also available on [Hugging Face Hub](https://huggingface.co/ConvLab).

- [Installation](#installation)
- [Tutorials](#tutorials)
- [Unified Datasets](#Unified-Datasets)
- [Models](#models)
- [Code Structure]($Code-Structure)
- [Contributing](#contributing)
- [Team](#Team)
- [Citing](#citing)
- [License](#license)

## Updates

- **2022.11.30**: ConvLab-3 release.

## Installation

You can install ConvLab-3 in the following ways according to your need. Higher versions of `torch` and `transformers` may also work.

### Git clone and pip install in development mode (Recommend)

For the latest and most configurable version, we recommend installing ConvLab-3 in development mode.

Clone the newest repository:

```bash
git clone --depth 1 https://github.com/ConvLab/ConvLab-3.git
```

Install ConvLab-3 via pip:

```bash
cd ConvLab-3
pip install -e .
```

### Pip install from PyPI

To use ConvLab-3 as an off-the-shelf tool, you can install via:

```bash
pip install convlab
```

### Using Docker

We also provide [Dockerfile](https://github.com/ConvLab/ConvLab-3/blob/master/Dockerfile) for building docker. Basically it uses the `requirement.txt` and then installs ConvLab-3 in development mode.

```bash
# create image
docker build -t convlab .

# run container
docker run -dit convlab

# open bash in container
docker exec -it CONTAINER_ID bash
```

## Tutorials

- [Getting Started](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb) (Have a try on [Colab](https://colab.research.google.com/github/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb)!) 
- [Introduction to unified data format](https://github.com/ConvLab/ConvLab-3/tree/master/data/unified_datasets)
- [Utility functions for unified datasets](https://github.com/ConvLab/ConvLab-3/blob/master/convlab/util/unified_datasets_util.py)
- [How to add a new dataset](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Add_New_Model.md)
- How to add a new model
- [How to use RL toolkit](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Train_RL_Policies)
- [Interactive tool](https://github.com/thu-coai/ConvLab-2/blob/master/deploy) [[demo video]](https://youtu.be/00VWzbcx26E)

## Unified Datasets

Current datasets in unified data format: (DA-U/DA-S stands for user/system dialog acts)

| Dataset       | Dialogs | Goal               | DA-U               | DA-S               | State              | API result         | DataBase           |
| ------------- | ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Camrest       | 676     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    | :white_check_mark: |
| WOZ 2.0       | 1200    |                    | :white_check_mark: |                    | :white_check_mark: |                    |                    |
| KVRET         | 3030    |                    | :white_check_mark: |                    | :white_check_mark: | :white_check_mark: |                    |
| DailyDialog   | 13118   |                    | :white_check_mark: |                    |                    |                    |                    |
| Taskmaster-1  | 13175   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |                    |
| Taskmaster-2  | 17303   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |                    |
| MultiWOZ 2.1  | 10438   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    | :white_check_mark: |
| Schema-Guided | 22825   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| MetaLWOZ      | 40203   | :white_check_mark: |                    |                    |                    |                    |                    |
| CrossWOZ (zh) | 6012    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Taskmaster-3  | 23757   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |

Unified datasets are available under `data/unified_datasets` directory as well as [Hugging Face Hub](https://huggingface.co/ConvLab). We will continue adding more datasets listed in [this issue](https://github.com/ConvLab/ConvLab-3/issues/11). If you want to add a listed/custom dataset to ConvLab-3, you can create an issue for discussion and then create pull-request. We will list you as the [contributors](#Team) and highly appreciate your contributions!

## Models

We list newly integrated models in ConvLab-3 that support unified data format and obtain strong performance. You can follow the link for more details about these models. Other models can be used in the same way as in ConvLab-2.

| Task                           | Models                                                       | Input           | Output           |
| ------------------------------ | ------------------------------------------------------------ | --------------- | ---------------- |
| Response Generation            | [T5](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5) | Context         | Response         |
| Goal-to-Dialog                 | [T5](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5) | Goal            | Dialog           |
| Natural Language Understanding | [T5](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5), [BERTNLU](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/nlu/jointBERT), [MILU](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/nlu/milu) | Context         | DA-U             |
| Dialog State Tracking          | [T5](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5), SUMBT, SetSUMBT, TripPy | Context         | State            |
| RL Policy                      | DDPT, PPO, PG                                                | State, DA-U, DB | DA-S             |
| Natural Language Generation    | [T5](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5), SC-GPT | DA-S            | Response         |
| End-to-End                     | SOLOIST                                                      | Context, DB     | State, Response  |
| User simulator                 | TUS, GenTUS                                                  | Goal, DA-S      | DA-U, (Response) |

Trained models are available on [Hugging Face Hub](https://huggingface.co/ConvLab).

## Code structure



## Contributing

We welcome contributions from community. Please see issues to find what we need.

- If you want to add a new dataset, model, or other feature, please describe the dataset/model/feature in an issue before creating pull-request.
- Small change like fixing a bug can be directly made by a pull-request.

## Team

**ConvLab-3** is maintained and developed by [Tsinghua University Conversational AI](http://coai.cs.tsinghua.edu.cn/) group (THU-COAI), the [Dialogue Systems and Machine Learning Group](https://www.cs.hhu.de/en/research-groups/dialog-systems-and-machine-learning.html) at Heinrich Heine University, Düsseldorf, Germany and Microsoft Research (MSR).

We would like to thank all contributors of ConvLab:

Yan Fang, Zhuoer Feng, Jianfeng Gao, Qihan Guo, Kaili Huang, Minlie Huang, Sungjin Lee, Bing Li, Jinchao Li, Xiang Li, Xiujun Li, Jiexi Liu, Lingxiao Luo, Wenchang Ma, Mehrad Moradshahi, Baolin Peng, Runze Liang, Ryuichi Takanobu, Dazhen Wan, Hongru Wang, Jiaxin Wen, Yaoqin Zhang, Zheng Zhang, Qi Zhu, Xiaoyan Zhu, Carel van Niekerk, Christian Geishauser, Hsien-chin Lin, Nurul Lubis, Xiaochen Zhu, Michael Heck, Shutong Feng, Milica Gašić.


## Citing

If you use ConvLab-3 in your research, please cite:

```

```

## License

Apache License 2.0
