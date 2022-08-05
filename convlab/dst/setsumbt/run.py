# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run"""

from transformers import BertConfig, RobertaConfig

from convlab.dst.setsumbt.utils import get_args


MODELS = {
    'bert': (BertConfig, "BertTokenizer"),
    'roberta': (RobertaConfig, "RobertaTokenizer")
}


def main():
    # Get arguments
    args, config = get_args(MODELS)

    if args.run_nbt:
        from convlab.dst.setsumbt.do.nbt import main
        main(args, config)
    if args.run_evaluation:
        from convlab.dst.setsumbt.do.evaluate import main
        main(args, config)


if __name__ == "__main__":
    main()
