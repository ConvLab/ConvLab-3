import json
import os
import sys

if __name__ == '__main__':
    merged_data = {'train': [], 'validation': [], 'test': []}
    print(sys.argv)
    for dataset_name in sys.argv[1:]:
        data_dir = os.path.join('data/dst', dataset_name, 'user/context_100')
        for data_split in merged_data:
            with open(os.path.join(data_dir, f'{data_split}.json'), 'r') as f:
                for line in f:
                    item = json.loads(line)
                    item['context'] = f"{dataset_name}: {item['context']}"
                    merged_data[data_split].append(item)
    for data_split in merged_data:
        data_dir = os.path.join('data/dst', '+'.join(sys.argv[1:]), 'user/context_100')
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, f'{data_split}.json'), 'w') as f:
            for item in merged_data[data_split]:
                f.write(json.dumps(item)+'\n')
