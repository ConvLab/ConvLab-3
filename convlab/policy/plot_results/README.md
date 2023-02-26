# Plot training curve

Plot training curve directly from tensorboard event files by using for instance

``` python plot_results/plot.py --dir=EXP_DIR --map-file=example_map.json --max-dialogues=MAX_DIALOGUES --out-file=plot_results/```

The structure of the map-file is like this:
```json
[
  {
    "dir": "dir_1",
    "legend": "Algorithm1"
  },
  {
    "dir": "dir_2",
    "legend": "Algorithm2"
  }
]
```
The value of `legend` will be the depicted in the plot legend.

The file structure of the exp_dir is like this:

    ├── exp_dir                  
        └── map["dir_1"]
            └── experiment_seed0*
                └── tb_dir
                    └── events.*
            └── experiment_seed1*
                └── tb_dir
                    └── events.* 
        └── map["dir_2"]
            └── experiment_seed0*
                └── tb_dir
                    └── events.*
            └── experiment_seed1*
                └── tb_dir
                    └── events.* 

If you want to truncate the figure to a certain number of training dialogues on the x-axis, use the argument `--max-dialogues`.

This script will automatically generate plots in the folder **--out-file** showing several performance metrics such as success rate and return, but also additional information such as the action distributions.