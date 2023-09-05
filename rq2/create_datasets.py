#!/usr/bin/env python

# This script creates train.json, valid.json, and test.json by filtering repos from Github
import os

LANGUAGE = os.environ.get("LANGUAGE", "TypeScript")
assert LANGUAGE in ["Java", "Python", "JavaScript", "TypeScript", "Go"]
MIN_PULL_REQUESTS = 50
MIN_COMMIT_COUNT = 50
MIN_USERS = 5
PUSHED_SINCE = "2016-01-01"
TIME_LIMIT_PER_REPO = 1800 # seconds
LIMIT_BY_LICENSE = True
MIN_STARS = 50

REPO_METADATA_URL = "https://file2.pelmers.com/repo_metadata.json"

# How often to pickle intermediate results
SAVE_EVERY = 60
JUST_CLEAN = True

# If true, then we only include examples where the comment is changed but not the code (thus no 0 labels)
# If false, then we follow the previous work and look for code changes, then 1 if comment is changed, 0 otherwise
FIXED_COMMENTS_ONLY = False

# If debug, then we only test on the first DEBUG_COUNT repos
DEBUG = True
DEBUG_COUNT = 120
# Repos that don't produce very good examples
BLOCKED_REPOS = ['vmware/alb-sdk']

from functools import lru_cache
from itertools import chain
import sys, subprocess, json, tempfile, argparse, pickle, re

from multiprocessing import Pool

from tqdm import tqdm

dirname = os.path.abspath(os.path.dirname(__file__))
r = lambda p: os.path.join(dirname, *p.split("/"))

# Define function x that given a command string, runs it with subprocess and streams the output
def x(cmd):
    return subprocess.run(cmd.split(" "), env={"GIT_TERMINAL_PROMPT": "0"}).returncode


def language_to_extension(lang):
    if lang == "Java":
        return ".java"
    if lang == "Python":
        return ".py"
    if lang == "JavaScript":
        return ".js"
    if lang == "TypeScript":
        return ".ts"
    if lang == "Go":
        return ".go"
    raise Exception(f"Unknown language {lang}")


def work_by_repo(repo):
    from parsers import create_labeled_dataset_from_git_repo

    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone the repo to a temporary directory
        print(f"Working on {repo}...")
        try:
            assert x(f"git clone https://github.com/{repo}.git {tmpdir}") == 0
        except AssertionError:
            print(f"Failed to clone {repo}")
            return repo, []
        # Add a time limit because we got to keep moving
        try:
            return repo, create_labeled_dataset_from_git_repo(
                tmpdir, language_to_extension(LANGUAGE), time_limit=TIME_LIMIT_PER_REPO, fixed_comments_only=FIXED_COMMENTS_ONLY
            )
        except Exception as e:
            print(f"Failed to parse {repo}: {e}")
            return repo, []


def cleanup_by_repo(repo_item):
    repo, examples = repo_item
    return repo, filter_for_changed_returns(cleanup_dataset(examples), LANGUAGE.lower())


def filter_for_changed_returns(all_examples, lang):
    '''Filter out examples where the return type or value is not changed, return new list'''
    if FIXED_COMMENTS_ONLY:
        return all_examples
    from parsers import get_language_and_parser, comments_fuzzy_equal
    language, parser = get_language_and_parser(lang)
    kept_examples = []
    if lang == 'java':
        return_statement_query = language.query('((return_statement) @ret)')
        return_type_query = language.query('((method_declaration type: (_) @typ))')
    elif lang == 'python' or lang == 'javascript' or lang == 'typescript':
        return_statement_query = language.query('((return_statement) @ret)')
    elif lang == 'go':
        return_statement_query = language.query('((return_statement) @ret)')
        func_return_query = language.query('((function_declaration result: (_) @res))')
        method_return_query = language.query('((method_declaration result: (_) @res))')
    def get_return_codes(code):
        if lang == 'java':
            # Wrap code with a class so it parses
            code = f'class A {{ {code} }}'
            tree = parser.parse(bytes(code, 'utf-8'))
            node = tree.root_node
            returns = '\n'.join([statement.text.decode() for statement, _ in return_statement_query.captures(node)])
            ret_typ = ' '.join([typ.text.decode() for typ, _ in return_type_query.captures(node)])
            return ret_typ + '\n' + returns
        elif lang == 'python' or lang == 'javascript' or lang == 'typescript':
            tree = parser.parse(bytes(code, 'utf-8'))
            node = tree.root_node
            # Python and JS don't have a return type, just join the return statements
            return '\n'.join([statement.text.decode() for statement, _ in return_statement_query.captures(node)])
        elif lang == 'go':
            tree = parser.parse(bytes(code, 'utf-8'))
            node = tree.root_node
            returns = '\n'.join([statement.text.decode() for statement, _ in return_statement_query.captures(node)])
            ret_typ = (
                ' '.join([res.text.decode() for res, _ in func_return_query.captures(node)]) or
                ' '.join([res.text.decode() for res, _ in method_return_query.captures(node)])
            )
            return ret_typ + '\n' + returns
        else:
            return ''
    for d in all_examples:
        old_returns = get_return_codes(d['old_code_raw'])
        new_returns = get_return_codes(d['new_code_raw'])
        if not comments_fuzzy_equal(old_returns, new_returns):
            kept_examples.append(d)
    return kept_examples


def cleanup_dataset(all_examples):
    '''
    Returned cleaned datasets, where we remove duplicate code/comment pairs
    and possibly generated code
    '''
    from parsers import comments_fuzzy_equal
    # Only accept comments that are mostly letters (a-zA-Z0-9 percentage greater than this)
    comment_letters_threshold = 0.9
    cleaned = []
    for d in all_examples:
        if 'maybe_generated' in d and d['maybe_generated'] or 'generated' in d['path']:
            continue
        # if the change is just that it's deprecated, how could the model know?
        if 'deprecated' in d['new_code_raw'].lower() and 'deprecated' not in d['old_code_raw'].lower():
            continue
        if 'deprecated' in d['new_comment_raw'].lower() and 'deprecated' not in d['old_comment_raw'].lower():
            continue
        # if the label is 1 but the comments are almost the same, then it's not a good example
        if d['label'] == 1 and comments_fuzzy_equal(d['old_comment_raw'], d['new_comment_raw'], fuzziness=3):
            continue
        # if somehow any comment is empty, then it's not a good example
        if not d['old_comment_raw'].strip() or not d['new_comment_raw'].strip():
            continue
        # Only keep comments that start with a capital letter
        if not d['old_comment_raw'][0].isupper() or not d['new_comment_raw'][0].isupper():
            continue
        # If it's an added todo, then it's not a good example
        if any([debt in d['new_comment_raw'].lower() for debt in ['todo', 'fixme']]):
            continue
        if any([repo in d['repo_url'] for repo in BLOCKED_REPOS]):
            continue
        # If the comment is not mostly letters, then it might be commented code
        n_comment_letters = sum([1 for c in d['old_comment_raw'] if c.isalpha() or c.isspace() or c.isdigit()])
        if n_comment_letters < comment_letters_threshold * len(d['old_comment_raw']):
            continue
        # Delete code lines in d that start with an @ (FIXME: why am I doing this?)
        d['old_code_raw'] = '\n'.join([l for l in d['old_code_raw'].split('\n') if not l.startswith('@')])
        d['new_code_raw'] = '\n'.join([l for l in d['new_code_raw'].split('\n') if not l.startswith('@')])
        cleaned.append(d)

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        help="Path to repos.pkl from a previous run",
        default=None,
    )
    args = parser.parse_args()

    # First change directory to ../notebooks
    os.chdir(r("../notebooks"))
    sys.path.append("lib")
    # Download repo metadata if it does not exist
    if not os.path.exists("data/repo_metadata.json"):
        x(f"wget {REPO_METADATA_URL} -O data/repo_metadata.json")

    # Pool size = # processors * 2 since the git stuff is IO bound
    if not DEBUG:
        p_count = max(4, 2 * os.cpu_count())
        mp = Pool(p_count)
        print(f"Using {p_count} processes for Pool")
    import pandas as pd
    DATASET_NAME = f"{LANGUAGE}-{pd.Timestamp.today().strftime('%Y-%m-%d')}"

    @lru_cache(maxsize=1)
    def get_repo_list():
        print("Reading data from repo_metadata.json")

        df = pd.read_json("data/repo_metadata.json")
        print(f"Data loaded from {REPO_METADATA_URL}, {len(df)} repos")

        # Filter repos based on constants at the top of the file
        lang_df = df[
            (df["pullRequests"] >= MIN_PULL_REQUESTS)
            & (df["primaryLanguage"] == LANGUAGE)
            & (df["pushedAt"] > PUSHED_SINCE)
            & (df["assignableUserCount"] >= MIN_USERS)
            & (df["defaultBranchCommitCount"] >= MIN_COMMIT_COUNT)
            & (df["stars"] >= MIN_STARS)
        ]
        if LIMIT_BY_LICENSE:
            lang_df = lang_df[(
                (lang_df["license"] == "MIT License")
                | (lang_df["license"] == 'BSD 3-Clause "New" or "Revised" License')
                | (lang_df["license"] == "Apache License 2.0")
            )]
        license_text = 'with MIT/Apache/BSD license' if LIMIT_BY_LICENSE else 'with any license'
        print(
            f"Number of {LANGUAGE} repos {license_text}, {MIN_PULL_REQUESTS} pull requests, pushed since {PUSHED_SINCE}, {MIN_USERS}+ assignable users, {MIN_COMMIT_COUNT}+ commits and {MIN_STARS}+ stars: {len(lang_df)}"
        )

        # Sorted by defaultBranchCommitCount
        lang_df = lang_df.sort_values("defaultBranchCommitCount", ascending=True)
        # Note: the deepjit dataset uses about 15 examples per repo (8398 training, 556 repos)
        if DEBUG:
            return lang_df["nameWithOwner"].tolist()[:DEBUG_COUNT]
        else:
            return lang_df["nameWithOwner"].tolist()


    def get_data(all_repos):
        for repo in examples_by_repo:
            all_repos.remove(repo)
        from parsers import build_shared_library
        build_shared_library()
        if DEBUG:
            generator = (work_by_repo(repo) for repo in all_repos)
        else:
            generator = mp.imap_unordered(work_by_repo, all_repos)
        # For each repo, clone it, parse the directory for examples
        for repo, examples in tqdm(generator, total=len(all_repos)):
            examples_by_repo[repo] = examples
            print(f'Done with {repo}, {len(examples)} examples, {len(examples_by_repo)} repos total')
            if len(examples_by_repo) % SAVE_EVERY == 0:
                with open(pickle_file, "wb") as f:
                    pickle.dump((examples_by_repo, metadata), f)
                    print(f"Results pickled to {pickle_file}, use --resume {pickle_file} to resume")

        with open(pickle_file, "wb") as f:
            pickle.dump((examples_by_repo, metadata), f)
            print(f"Done. Results pickled to {pickle_file}.")

    def clean():
        # Now clean the data and save in a new file pickle_file.replace("repos", "repos-cleaned")
        total_examples = sum([len(d) for d in examples_by_repo.values()])
        print(f'Cleaning the data ({total_examples} examples)...')

        if DEBUG:
            generator = (cleanup_by_repo(item) for item in examples_by_repo.items())
        else:
            generator = mp.imap_unordered(cleanup_by_repo, examples_by_repo.items())
        for repo, cleaned_ex in tqdm(generator, total=len(examples_by_repo)):
            examples_by_repo[repo] = cleaned_ex

        # Compute the top/bottom by dataset length, and remove those
        top_removal_threshold = 0.10
        bottom_removal_threshold = 0.00
        sample_lengths = sorted([len(d['old_comment_raw']) + len(d['new_code_raw']) for d in chain.from_iterable(examples_by_repo.values())])
        print(f"Using total sample lengths: len={len(sample_lengths)}, min={min(sample_lengths)}, max={max(sample_lengths)}")
        top_length = sample_lengths[int((len(sample_lengths) - 1) * (1-top_removal_threshold))]
        bottom_length = sample_lengths[int(len(sample_lengths) * bottom_removal_threshold)]
        print(f"Removing duplicates and samples with length > {top_length} or < {bottom_length}...")
        # Globally filter for duplicates of old_comment + new_code
        seen_examples = set()
        for repo, examples in tqdm(examples_by_repo.items()):
            new_examples = []
            for example in examples:
                if (len(example["old_comment_raw"]) + len(example["new_code_raw"]) <= top_length and
                    len(example["old_comment_raw"]) + len(example["new_code_raw"]) >= bottom_length):
                    if example["old_comment_raw"] + example["new_code_raw"] not in seen_examples:
                        new_examples.append(example)
                        seen_examples.add(example["old_comment_raw"] + example["new_code_raw"])
                    else:
                        if DEBUG:
                            print('found duplicate')
            examples_by_repo[repo] = new_examples

        sample_lengths = [len(d['old_comment_raw']) + len(d['new_code_raw']) for cleaned in examples_by_repo.values() for d in cleaned]
        print(f"After removal sample lengths: len={len(sample_lengths)}, min={min(sample_lengths)}, max={max(sample_lengths)}")
        total_positive = sum([d['label'] for cleaned in examples_by_repo.values() for d in cleaned])
        print(f"Total positive examples: {total_positive}")
        # Delete all the keys that have no examples
        for repo in list(examples_by_repo.keys()):
            if len(examples_by_repo[repo]) == 0:
                del examples_by_repo[repo]
        print(f"Removed {metadata['NUM_REPOS_IN_FILTER'] - len(examples_by_repo)} repos with no examples")
        # Save
        with open(pickle_file.replace("repos", "repos-cleaned"), "wb") as f:
            pickle.dump((examples_by_repo, metadata), f)
            print(f"Done. Results pickled to {pickle_file.replace('repos', 'repos-cleaned')}.")

    if args.resume:
        args.resume = args.resume.replace("notebooks/", "")
        with open(args.resume, "rb") as f:
            examples_by_repo, _ = pickle.load(f)
            print(f"Loaded {len(examples_by_repo)} from previous run")
    else:
        examples_by_repo = {}

    all_repos = get_repo_list()
    pickle_file = f"data/repos-{DATASET_NAME}-{len(all_repos)}.pkl"
    metadata = {
        "LANGUAGE": LANGUAGE,
        "MIN_PULL_REQUESTS": MIN_PULL_REQUESTS,
        "MIN_COMMIT_COUNT": MIN_COMMIT_COUNT,
        "MIN_STARS": MIN_STARS,
        "MIN_USERS": MIN_USERS,
        "PUSHED_SINCE": PUSHED_SINCE,
        "REPO_METADATA_URL": REPO_METADATA_URL,
        "NUM_REPOS_IN_FILTER": len(all_repos),
    }
    if JUST_CLEAN:
        clean()
    else:
        get_data(all_repos)
        clean()


if __name__ == "__main__":
    main()