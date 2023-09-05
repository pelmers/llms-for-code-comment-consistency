#!/usr/bin/env python

# example-reader: let you read examples of train/valid/test json files one by one
# input: filename of the data file to look at

import sys
import argparse
import json
import random

from resample_dataset import print_example_statistics

def read_example(example):
    """Read an example"""
    old_comment_raw = example['old_comment_raw']
    old_code_raw = example['old_code_raw']
    label = 'good' if example['label'] == 0 else 'bad'
    comment_char = '// '
    if 'path' in example and example['path'].endswith('.py'):
        comment_char = '# '
    if 'repo_url' in example:
        print(f"Repo: {example['repo_url']}")
    if 'path' in example:
        print(f"Path: {example['path']}")
    if 'commit' in example:
        print(f"Commit: {example['commit']}")
    elif 'id' in example:
        print(f"ID: {example['id']}")
    print(f"{comment_char}Old comment ({label}):")
    print(comment_char + old_comment_raw)
    print(comment_char + 'Old code:\n' + old_code_raw)

    new_comment_raw = example['new_comment_raw']
    new_code_raw = example['new_code_raw']
    print(f"{comment_char}New comment:")
    print(comment_char + new_comment_raw)
    print(comment_char + 'New code:\n' + new_code_raw)


# Write hypothesis tester that lets me count how many GOOD comments were actually bad
def test_hypothesis(examples, limit):
    """Test hypothesis that all good comments are actually bad"""
    tested = 0
    correct = 0
    for ex in examples:
        if ex['label'] == 0:
            read_example(ex)
            checked_label = input(f"Is this comment good? (y/n) ")
            if checked_label.lower() == 'y':
                correct += 1
            tested += 1
        if tested >= limit:
            break
    print(f"Accuracy: {correct / len(examples)}")
    print(f"Tested {tested} examples")
    print(f"Correct {correct} examples")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Read examples of train/valid/test json files one by one')
    parser.add_argument('filename', help='filename of the data file to look at')
    # Add integer argument, if present then test hypothesis
    parser.add_argument('--test', type=int, help='test hypothesis that all good comments are actually bad')

    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        examples = json.load(f)
        print_example_statistics(examples)
        random.shuffle(examples)
        # sort examples by length of new_code_raw
        examples.sort(key=lambda x: len(x['new_code_raw']))
        if args.test:
            test_hypothesis(examples, args.test)
        else:
            for ex in examples[:]:
                read_example(ex)
                pass
                # if not in the debugger, then wait for user input
                if not 'debugpy.common' in sys.modules:
                    input("")

if __name__ == '__main__':
    main()