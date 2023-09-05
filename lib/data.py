import json

import torch
from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, ids, labels, indices):
        self.ids = ids
        self.labels = labels
        self.indices = indices
        assert len(indices) == len(ids) == len(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx], self.indices[idx]


class CustomConcatDataset(CustomDataset):
    def __init__(self, ds1, ds2):
        super().__init__(ds1.ids + ds2.ids, ds1.labels + ds2.labels, ds1.indices + ds2.indices)


def custom_dataset_from_zipped_items(zipped_items):
    ids = []
    labels = []
    indices = []
    for item in zipped_items:
        ids.append(item[0])
        labels.append(item[1])
        indices.append(item[2])
    return CustomDataset(ids, labels, indices)


class CustomPostHocDataset(Dataset):
    def __init__(self, filename, model_type, tokenizer, max_length):
        self.entries = dataset_from_file(filename, model_type, tokenizer, max_length, show_plots=False)[0]
        self.old_code_dataset = dataset_from_file(filename, model_type, tokenizer, max_length, show_plots=False, use_old_code=True)[1]
        self.new_code_dataset = dataset_from_file(filename, model_type, tokenizer, max_length, show_plots=False, use_old_code=False)[1]
        assert len(self.old_code_dataset) == len(self.new_code_dataset)

    def __len__(self):
        return len(self.old_code_dataset)

    def __getitem__(self, idx):
        label = self.new_code_dataset[idx][1]
        # Return old ids, new ids, indices, label as tuple
        return (
            self.old_code_dataset[idx][0],
            self.new_code_dataset[idx][0],
            self.new_code_dataset[idx][2],
            label
        )


# Function to load benchmark dataset
# benchmark is a regular text file with the format
# ###### n ######
# URL: url
# Review: commit message
# Old Version: old comment + old code
# New Version: new comment + new code
def dataset_from_benchmark_file(filename, model_type, tokenizer, max_length, show_plots=True):
    pass


def dataset_from_file(filename, model_type, tokenizer, max_length, show_plots=True, use_old_code=False):
    all_indices = []
    all_ids = []
    all_labels = []
    n_truncated = 0
    combined_lengths = []
    def to_tensors(index, entry):
        nonlocal n_truncated
        # tokenize new comment and new code
        # TODO: do prompt tuning by putting in some extra stuff?
        code = entry['old_code_raw'] if use_old_code else entry['new_code_raw']
        if model_type == 'codebert':
            tok = [tokenizer.cls_token] + tokenizer.tokenize(entry['old_comment_raw']) + \
                  [tokenizer.sep_token] + tokenizer.tokenize(code) + \
                  [tokenizer.eos_token]
            if len(tok) > max_length:
                n_truncated += 1
                tok = tok[:max_length]
        else:
            comment_char = '// '
            if 'path' in entry and entry['path'].endswith('.py'):
                comment_char = '# '
            tok = tokenizer.tokenize(comment_char + entry['old_comment_raw'] + '\n' + code)
            prompt_tok = tokenizer.tokenize('\n' + tokenizer.eos_token)
            if len(tok) + len(prompt_tok) > max_length:
                n_truncated += 1
                tok = tok[:max_length - len(prompt_tok)] + prompt_tok
        combined_lengths.append(len(tok))

        ids = tokenizer.convert_tokens_to_ids(tok)

        all_indices.append(index)
        all_ids.append(ids)
        all_labels.append(entry['label'])

    # Read the file as json and store each entry in a list
    # Each entry is a dictionary with the following keys
    # id, label, comment_type, old_code_raw, new_code_raw, old_comment_raw, new_comment_raw
    # label = 0 if good, 1 if bad
    # bad means new comment and new code do not match, good means they do
    data_entries = []
    with open(filename) as f:
        data = json.load(f)
        print('Loading {} examples from {}'.format(len(data), filename))
        for entry in tqdm(data):
            to_tensors(len(data_entries), entry)
            data_entries.append(entry)

    if len(data_entries) == 0:
        return [], CustomDataset([], [], [])

    n_total_tokens = sum(combined_lengths)
    print('Total number of tokens: {}'.format(len(data_entries) * max_length))
    print('Total number of truncated examples: {}'.format(n_truncated))
    print('Total number of input tokens (ignoring truncation): {}'.format(n_total_tokens))
    # Print basic statistics of combined lengths list with numpy
    print('Mean: {}'.format(np.mean(combined_lengths)))
    print('Median: {}'.format(np.median(combined_lengths)))
    print('Standard deviation: {}'.format(np.std(combined_lengths)))
    print('Minimum: {}'.format(np.min(combined_lengths)))
    print('Maximum: {}'.format(np.max(combined_lengths)))

    if show_plots:
        import matplotlib.pyplot as plt
        # Create box plot of comment, code, and combined lengths with plt
        plt.boxplot([combined_lengths], labels=['combined'])
        plt.title('Token lengths')
        plt.show()
        # Also show a histogram of the combined lengths, bucket all lengths over 90 percentile into one bucket
        # Use percentages on the y axis
        plt.hist(combined_lengths, bins=16, range=(0, 512), weights=np.ones(len(combined_lengths)) / len(combined_lengths))
        from matplotlib.ticker import PercentFormatter
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.title('Tokenized lengths')
        plt.show()

    return data_entries, CustomDataset(all_ids, all_labels, all_indices)


def get_collate_fn(tokenizer):
    padding_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

    def collate_dataset_entries(data):
        """Data is a list of tuples (token_ids list, label number, index number)"""
        all_labels = torch.tensor([entry[1] for entry in data])
        all_indices = torch.tensor([entry[2] for entry in data])
        # Here we pad all the tensors to the longest entry's length
        longest_sample_length = max([len(entry[0]) for entry in data])
        padded_ids = torch.ones((len(data), longest_sample_length), dtype=torch.long) * padding_token_id
        attention_masks = torch.zeros((len(data), longest_sample_length), dtype=torch.long)
        for i, entry in enumerate(data):
            ids, _, _ = entry
            padded_ids[i, :len(ids)] = torch.tensor(entry[0])
            attention_masks[i, :len(ids)] = torch.tensor([1] * len(entry[0]))
        return padded_ids, attention_masks, all_labels, all_indices

    return collate_dataset_entries


def get_posthoc_collate_fn(tokenizer):
    padding_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

    def collate_dataset_entries(data):
        """Data is a list of tuples (old ids, new ids, index, label)"""
        # Return a tuple of (old_ids, old_attention_masks, new_ids, new_attention_masks, labels, indices)
        all_labels = torch.tensor([entry[3] for entry in data])
        all_indices = torch.tensor([entry[2] for entry in data])
        # Take the longest length as the longest new or old id vector
        longest_sample_length = max([max(len(entry[0]), len(entry[1])) for entry in data])
        padded_old_ids = torch.ones((len(data), longest_sample_length), dtype=torch.long) * padding_token_id
        padded_new_ids = torch.ones((len(data), longest_sample_length), dtype=torch.long) * padding_token_id
        old_attention_masks = torch.zeros((len(data), longest_sample_length), dtype=torch.long)
        new_attention_masks = torch.zeros((len(data), longest_sample_length), dtype=torch.long)
        for i, entry in enumerate(data):
            old_ids, new_ids, _, _ = entry
            padded_old_ids[i, :len(old_ids)] = torch.tensor(entry[0])
            old_attention_masks[i, :len(old_ids)] = torch.tensor([1] * len(entry[0]))
            padded_new_ids[i, :len(new_ids)] = torch.tensor(entry[1])
            new_attention_masks[i, :len(new_ids)] = torch.tensor([1] * len(entry[1]))
        return padded_old_ids, old_attention_masks, padded_new_ids, new_attention_masks, all_labels, all_indices

    return collate_dataset_entries
