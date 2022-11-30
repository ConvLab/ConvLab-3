## NLU benchmark for BERTNLU and MILU on multiwoz21, tm1, tm2, tm3

To illustrate that it is easy to use the model for any dataset that in our unified format, we report the performance on several datasets in our unified format. We follow `README.md` and config files in `unified_datasets/` to generate `predictions.json`, then evaluate it using `../evaluate_unified_datasets.py`. Note that we use almost the same hyper-parameters for different datasets, which may not be optimal.

<table>
<thead>
  <tr>
    <th></th>
    <th colspan=2>MultiWOZ 2.1</th>
    <th colspan=2>Taskmaster-1</th>
    <th colspan=2>Taskmaster-2</th>
    <th colspan=2>Taskmaster-3</th>
  </tr>
</thead>
<thead>
  <tr>
    <th>Model</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>T5-small</td>
    <td>77.8</td><td>86.5</td>
    <td>74.0</td><td>52.5</td>
    <td>80.0</td><td>71.4</td>
    <td>87.2</td><td>83.1</td>
  </tr>
  <tr>
    <td>T5-small (context=3)</td>
    <td>82.0</td><td>90.3</td>
    <td>76.2</td><td>56.2</td>
    <td>82.4</td><td>74.3</td>
    <td>89.0</td><td>85.1</td>
  </tr>
  <tr>
    <td>BERTNLU</td>
    <td>74.5</td><td>85.9</td>
    <td>72.8</td><td>50.6</td>
    <td>79.2</td><td>70.6</td>
    <td>86.1</td><td>81.9</td>
  </tr>
  <tr>
    <td>BERTNLU (context=3)</td>
    <td>80.6</td><td>90.3</td>
    <td>74.2</td><td>52.7</td>
    <td>80.9</td><td>73.3</td>
    <td>87.8</td><td>83.8</td>
  </tr>
  <tr>
    <td>MILU</td>
    <td>72.9</td><td>85.2</td>
    <td>72.9</td><td>49.2</td>
    <td>79.1</td><td>68.7</td>
    <td>85.4</td><td>80.3</td>
  </tr>
  <tr>
    <td>MILU (context=3)</td>
    <td>76.6</td><td>87.9</td>
    <td>72.4</td><td>48.5</td>
    <td>78.9</td><td>68.4</td>
    <td>85.1</td><td>80.1</td>
  </tr>
</tbody>

- Acc: whether all dialogue acts of an utterance are correctly predicted
- F1: F1 measure of the dialogue act predictions over the corpus.