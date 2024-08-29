# Format data for erasing
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
            prog='format_input',
            description='Generate input for erasing repo (https://github.com/rohitgandikota/erasing.git)')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='./datasets/erasing_dataset.csv')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = pd.read_csv(args.dataset)
    if not ('case_number' in dataset.columns):
        print('Adding case numbers...')
        case_numbers = np.arange(len(dataset))
        dataset.insert(0, 'case_number', case_numbers)

    if not ('evaluation_seed' in dataset.columns):
        print('Adding evaluation seeds...')
        eval_seeds = np.random.randint(0, 10000, size=len(dataset))
        dataset.insert(0, 'evaluation_seed', eval_seeds)

    print(dataset)
    print(f'Saving dataset to {args.output}...')
    dataset.to_csv(args.output)
