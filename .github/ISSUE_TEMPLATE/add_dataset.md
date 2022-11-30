---
name: Add dataset
about: Add a new dataset in unified format
title: "[Dataset] "
labels: dataset, new
assignees: ''

---

- **Name:** *name of the dataset*
- **Description:** *short description of the dataset (or link to social media or blog post)*
- **Paper:** *link to the dataset paper if available*
- **Data:** *link to the Github repository or current dataset location*
- **License:** *what is the license of the dataset*
- **Motivation:** *what are some good reasons to have this dataset*


### Checkbox

- [ ] Create `data/unified_datasets/$dataset_name` folder, where `$dataset_name` is the name of the dataset.
- [ ] Create the dataset scripts under `data/unified_datasets/$dataset_name` following `data/unified_datasets/README.md`.
- [ ] Run `python check.py $dataset` in the `data/unified_datasets` directory to check the validation of processed dataset and get data statistics and shuffled dialog ids.
- [ ] Add the dataset card `data/unified_datasets/$dataset_name/README.md` following `data/unified_datasets/README_TEMPLATE.md`.
- [ ] Upload the data, scripts, and dataset card to https://huggingface.co/ConvLab
- [ ] Update `NOTICE` with license information.
