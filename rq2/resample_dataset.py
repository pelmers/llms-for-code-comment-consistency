#!/usr/bin/env python

# This script takes as input a folder containing train, valid, test.json and metadata
# Given the parameters set at the top of the file, we produce a new train, valid, test.json

# Total size of test + train + valid sets
TOTAL_SIZE = 22_000
# Whether to make number of positive and negative examples equal
BALANCE = True
TRAIN_SPLIT = [0.8, 0.1, 0.1]

import argparse, json, os, pickle, subprocess, re
import random
from tqdm import tqdm
import sys

assert sum(TRAIN_SPLIT) == 1.0

dirname = os.path.abspath(os.path.dirname(__file__))
r = lambda p: os.path.join(dirname, *p.split("/"))


def read_dataset(data_file):
    """Read all examples from pickled dataset file, return as tuple (examples, metadata)"""
    with open(data_file, "rb") as f:
        examples_by_repo, metadata = pickle.load(f)
        print(f"Loaded {len(examples_by_repo)} from previous run")
        all_examples = []
        for examples in examples_by_repo.values():
            all_examples.extend(examples)
        return all_examples, metadata


def reduce_dataset(all_examples, size, return_extras=False):
    """Reduce dataset size to specified TOTAL_SIZE, rebalance if necessary."""
    # If BALANCE is True, then we make sure that the number of positive and negative examples are equal
    # If BALANCE is False, then we just take the first TOTAL_SIZE examples after shuffling
    # First, shuffle the dataset
    random.shuffle(all_examples)
    size = int(size)
    # Then, take the first size examples
    if not BALANCE:
        if return_extras:
            return all_examples[:size], all_examples[size:]
        else:
            return all_examples[:size]
    else:
        # If we're balancing, we need to make sure that the number of positive and negative examples are equal
        # We do this by taking the first size/2 positive examples and the first size/2 negative examples
        # If size is odd, we take the extra example from the positive examples
        pos_examples = [ex for ex in all_examples if ex['label'] == 1]
        neg_examples = [ex for ex in all_examples if ex['label'] == 0]
        reduced_set = pos_examples[:size//2] + neg_examples[:size//2 + size%2]
        if return_extras:
            return reduced_set, pos_examples[size//2:] + neg_examples[size//2 + size%2:]
        else:
            return reduced_set


def split_dataset(all_examples):
    """Split dataset into train, valid, test, where each repo is in one of the splits"""
    examples_by_repo = {}
    for ex in tqdm(all_examples):
        repo = '/'.join(ex['repo_url'].split('/')[-2:]).replace('.git', '')
        if repo not in examples_by_repo:
            examples_by_repo[repo] = []
        examples_by_repo[repo].append(ex)

    # At the end, write out train.json, valid.json, and test.json, organize the data so each repo is in one of the splits
    total_number_of_examples = sum(
        [len(examples) for examples in examples_by_repo.values()]
    )
    print(f"Total of {total_number_of_examples} examples")
    train_split_size = int(TRAIN_SPLIT[0] * total_number_of_examples)
    valid_split_size = int(TRAIN_SPLIT[1] * total_number_of_examples)
    test_split_size = int(TRAIN_SPLIT[2] * total_number_of_examples)
    # shuffle the list of repo names
    repo_names = list(examples_by_repo.keys())
    random.shuffle(repo_names)

    train_set = []
    valid_set = []
    test_set = []
    for repo in repo_names:
        if len(train_set) < train_split_size:
            train_set += examples_by_repo[repo]
        elif len(valid_set) < valid_split_size:
            valid_set += examples_by_repo[repo]
        elif len(test_set) < test_split_size:
            test_set += examples_by_repo[repo]

    return train_set, valid_set, test_set


import cdifflib

def non_space_diff_ratio(a, b):
    """Get distance between two strings, ignoring all whitespace"""
    a = re.sub(r'\s', '', a)
    b = re.sub(r'\s', '', b)
    return cdifflib.CSequenceMatcher(None, a, b).ratio()


def print_example_statistics(examples):
    # Print out some stats: the median/mean/min/max length of the comments, comments + code
    # The number of repositories represented and the number of examples per repo by label (top 10)
    def get_repo(ex):
        if 'repo_url' in ex:
            return '/'.join(ex['repo_url'].split('/')[-2:]).replace('.git', '')
        else:
            # ex['id'] format is owner_repo-hash-type-version
            # the owner/repo could have dashes or underscores in it, so start from the end
            rest = '-'.join(ex['id'].split('-')[:-3])
            owner = rest.split('_')[0]
            return f'{owner}/{rest[len(owner)+1:]}'
    repo_counts = {}
    for ex in examples:
        repo = get_repo(ex)
        if repo not in repo_counts:
            repo_counts[repo] = [0, 0]
        repo_counts[repo][ex['label']] += 1
    print(f"Number of examples: {len(examples)}, {sum([ex['label'] for ex in examples])} positive")
    print(f"Number of repos: {len(repo_counts)}")
    print(f"\nTop 10 repos by number of examples:")
    for repo, counts in sorted(repo_counts.items(), key=lambda x: sum(x[1]), reverse=True)[:10]:
        print(f"{repo}: {counts[0]} positive, {counts[1]} negative")
    print(f"\nTop 10 repos by number of positive examples:")
    for repo, counts in sorted(repo_counts.items(), key=lambda x: x[1][0], reverse=True)[:10]:
        print(f"{repo}: {counts[0]} positive, {counts[1]} negative")
    print(f"\nTop 10 repos by number of negative examples:")
    for repo, counts in sorted(repo_counts.items(), key=lambda x: x[1][1], reverse=True)[:10]:
        print(f"{repo}: {counts[0]} positive, {counts[1]} negative")
    # Length of example is old comment raw + new code raw
    lengths = [len(ex['old_comment_raw']) + len(ex['new_code_raw']) for ex in examples]
    print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
    print(f"Mean length: {sum(lengths) / len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    # Print old/new comment match ratio statistics for all positive examples
    ratios = []
    for ex in tqdm(examples):
        if ex['label'] == 1:
            ratios.append(non_space_diff_ratio(ex['old_comment_raw'], ex['new_comment_raw']))
    print(f"Median old/new comment match ratio: {sorted(ratios)[len(ratios) // 2]}")
    print(f"Mean old/new comment match ratio: {sum(ratios) / len(ratios)}")
    print(f"Min old/new comment match ratio: {min(ratios)}")
    print(f"Max old/new comment match ratio: {max(ratios)}")


def main():
    parser = argparse.ArgumentParser(description='Resample dataset')
    parser.add_argument('input_file', help='Path to dataset file of examples from create_datasets')
    parser.add_argument('output_folder', help='folder to output new train, valid, test.json and metadata')
    args = parser.parse_args()
    examples, metadata = read_dataset(args.input_file)

    print(f'Read {len(examples)} examples, language {metadata["LANGUAGE"]}, repos in filter: {metadata["NUM_REPOS_IN_FILTER"]}')
    print(f'Positive examples: {sum([ex["label"] for ex in examples])}')

    print_example_statistics(examples)

    train, valid, test = split_dataset(examples)
    print(f'After splitting, {len(train)} train, {len(valid)} valid, {len(test)} test examples')

    train, extras = reduce_dataset(train, TOTAL_SIZE * TRAIN_SPLIT[0], return_extras=True)
    print(f'After resampling train, {len(train)} examples with {sum([ex["label"] for ex in train])} positive')
    print(f'Extra examples: {len(extras)}, {sum([ex["label"] for ex in extras])} positive')
    valid = reduce_dataset(valid, TOTAL_SIZE * TRAIN_SPLIT[1])
    print(f'After resampling valid, {len(valid)} examples with {sum([ex["label"] for ex in valid])} positive')
    test = reduce_dataset(test, TOTAL_SIZE * TRAIN_SPLIT[2])
    print(f'After resampling test, {len(test)} examples with {sum([ex["label"] for ex in test])} positive')

    print(f"Printing train example statistics")
    print_example_statistics(train)
    print(f"Printing valid example statistics")
    print_example_statistics(valid)
    print(f"Printing test example statistics")
    print_example_statistics(test)

    # Write dataset to output folder, create if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    with open(os.path.join(args.output_folder, 'train.json'), 'w') as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(args.output_folder, 'valid.json'), 'w') as f:
        json.dump(valid, f, indent=2)
    with open(os.path.join(args.output_folder, 'test.json'), 'w') as f:
        json.dump(test, f, indent=2)
    with open(os.path.join(args.output_folder, 'extras.json'), 'w') as f:
        json.dump(extras, f, indent=2)
    with open(os.path.join(args.output_folder, 'metadata.json'), 'w') as f:
        metadata['UNSAMPLED_FILENAME'] = args.input_file
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
