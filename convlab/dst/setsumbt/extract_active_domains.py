# coding=utf-8
# --------------------------------------------------------------------------------
# Project: Training Project
# Author: Carel van Niekerk
# Year: 2024
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
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
# limitations under the License."
"""Extract active domains from the dataloaders."""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from convlab.dst.setsumbt.datasets.unified_format import UnifiedFormatDataset
from convlab.dst.setsumbt.datasets.utils import IdTensor


def load_data(path: Path) -> UnifiedFormatDataset:
    loader = torch.load(path)
    return loader.dataset


def extract_active_domains(
    data: UnifiedFormatDataset,
) -> dict[str, dict[str, list[str]]]:
    features = {
        key: itm
        for key, itm in data.features.items()
        if key == "dialogue_ids" or "active_domain" in key
    }

    active_domains = {}
    for i, dialogue_id in tqdm(enumerate(features["dialogue_ids"])):
        dialogue_id = dialogue_id[0]  # noqa: PLW2901
        _domains = {}
        for key, itm in features.items():
            if key == "dialogue_ids":
                continue
            domain = key.split("-")[-1]
            for turn_id, active in enumerate(itm[i]):
                if active != -1:
                    if str(turn_id + 1) not in _domains:
                        _domains[str(turn_id + 1)] = []
                    if active == 1:
                        _domains[str(turn_id + 1)].append(domain)
        active_domains[dialogue_id] = _domains

    return active_domains


def main() -> None:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Extract active domains from the dataloaders.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the model.",
    )
    args = parser.parse_args()

    for set_type in ["train", "dev", "test"]:
        print(f"Extracting active domains for {set_type} set.")
        data = load_data(args.model_path / "dataloaders" / f"{set_type}.dataloader")
        active_domains = extract_active_domains(data)

        target_path = (
            args.model_path / "dataloaders" / f"{set_type}_active_domains.json"
        )
        with target_path.open("w") as f:
            json.dump(active_domains, f, indent=4)


if __name__ == "__main__":
    main()
