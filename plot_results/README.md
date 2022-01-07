# Plot training curve

Plot training curve directly from tensorboard event files.

The file structure of tb_file is like this:

    .
    ├── dir                  
        └── map["dir"]
            └── experiment_*
                └── tb_dir
                    └── events.* 

The structure of map file is like this:
```json
[
  {
    "dir": "Algorithm1_results",
    "legend": "Algorithm1"
  },
  {
    "dir": "Algorithm2_results",
    "legend": "Algorithm2"
  }
]
```
The value of `legend` will be the depicted in the plot legend.

If you want to truncate the figure to a certain number of training dialogues on the x-axis, use the argument `--max-dialogues`.
