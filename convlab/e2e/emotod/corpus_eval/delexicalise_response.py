from convlab.util import load_dataset, create_delex_data

dataset_name = 'multiwoz21'
dataset = load_dataset(dataset_name)

# replace system utts in dataset with pipeline response



dataset , delex_vocab = create_delex_data(dataset)

# extract system response only