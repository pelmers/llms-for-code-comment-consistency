#!/usr/bin/env python

# mine-all-pr-comments: given a github repository in the form author/repo,
# download all of the pull requests comments and output them to stdout
# make sure to set GITHUB_TOKEN to your personal access token

import os
import sys
import json
import argparse
import requests
import time
from tqdm import tqdm
import pandas as pd
from functools import lru_cache
import multiprocessing as mp

GITHUB_TOKEN = open('github_token').read().strip()

MIN_STARS = 25
MIN_PULL_REQUESTS = 25
LANGUAGE = os.environ.get("LANGUAGE", "Java")

RUN_NAME = f"mined-comments-{MIN_STARS}stars-{MIN_PULL_REQUESTS}prs-{LANGUAGE}"
SAVE_FILE = f"data/{RUN_NAME}.json.gz"

IMPORTANT_KEYS = ['html_url', 'path', 'line', 'body', 'user', 'diff_hunk', 'author_association', 'commit_id', 'id']

SAVE_EVERY = 50
PROCESSOR_COUNT = 3

dirname = os.path.abspath(os.path.dirname(__file__))
r = lambda p: os.path.join(dirname, *p.split("/"))


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


EXTENSION = language_to_extension(LANGUAGE)


def get_pr_comments(repo, page_limit=100):
    """Get all the comments from all the pull requests of a repo"""
    url = lambda page: f'https://api.github.com/repos/{repo}/pulls/comments?per_page=100&page={page}'
    headers = {'Authorization': f'Bearer {GITHUB_TOKEN}', 'Accept': 'application/vnd.github+json'}
    page_number = 1
    all_comments = []
    fails = 0
    while True:
        page_url = url(page_number)
        try:
            response = requests.get(page_url, headers=headers)
            response.raise_for_status()
            comments = response.json()
            if not comments:
                break
            for c in comments:
                if 'user' in c and c['user'] is not None:
                    c['user'] = c['user']['login']
                else:
                    c['user'] = None
            comments = [{k: c[k] for k in IMPORTANT_KEYS} for c in comments]
            comments = [c for c in comments if 'path' in c and c['path'].endswith(EXTENSION)]
            all_comments.extend(comments)
            fails = 0
            page_number += 1
        except requests.exceptions.HTTPError as e:
            # If error code is 403 and 'Retry-After' header is given, then sleep for the specified seconds
            if response.status_code == 403 and 'Retry-After' in response.headers:
                print(f'Rate limit exceeded, sleeping for {response.headers["Retry-After"]} seconds')
                time.sleep(int(response.headers['Retry-After']))
                continue
            print('error', e, response.text, response.headers)
            fails += 1
            if fails > 3:
                print(f'too many fails, quitting {repo}')
                break
        if page_number > page_limit:
            print(f'page limit reached, quitting for {repo}')
            break
    return all_comments


def save_results(comment_map):
    import gzip
    with gzip.open(SAVE_FILE, 'wt') as f:
        json.dump(comment_map, f)
    print(f'Saved {len(comment_map)} repos with {sum(len(v) for v in comment_map.values())} comments to {SAVE_FILE}')


def load_results():
    import gzip
    with gzip.open(SAVE_FILE, 'rt') as f:
        print(f'Loading comments from {SAVE_FILE}')
        return json.load(f)


@lru_cache(maxsize=1)
def get_repo_list():
    '''Parses the repo list and applies filters, return a list of repositories.'''
    df = pd.read_json("data/repo_metadata.json")
    df = df[
            (df["pullRequests"] >= MIN_PULL_REQUESTS)
            & (df["primaryLanguage"] == LANGUAGE)
            & (df["stars"] >= MIN_STARS)
            & (df["isFork"] == False)
            & (df["isArchived"] == False)
        ]
    # Get a list of all df 'nameWithOwner' values, sorted by 'stars'
    repos = df.sort_values(by="stars", ascending=False)["nameWithOwner"].tolist()
    print(f'Obtained {len(repos)} repos from repo_metadata.json')
    return repos


def process_repo(repo):
    print(f'Loading comments for {repo}...')
    comments = get_pr_comments(repo)
    print(f'Loaded {len(comments)} comments for {repo}')
    return repo, comments


def main():
    parser = argparse.ArgumentParser()
    # Add a true/false flag to resume from a previous run
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # First change directory to ../notebooks
    os.chdir(r("../notebooks"))
    repos = get_repo_list()
    comment_map = load_results() if args.resume else {}
    if comment_map:
        print(f'Loaded {len(comment_map)} comments from previous run')
    # Remove any repos already in the comment map
    repos = [r for r in repos if r not in comment_map]
    generator = (process_repo(repo) for repo in repos)
    if PROCESSOR_COUNT > 1:
        pool = mp.Pool(PROCESSOR_COUNT)
        generator = pool.imap_unordered(process_repo, repos)
        print(f'Using {PROCESSOR_COUNT} processors')
    for repo, comments in tqdm(generator, total=len(repos)):
        comment_map[repo] = comments
        if len(comment_map) % SAVE_EVERY == 0:
            save_results(comment_map)

    save_results(comment_map)
    print(f'Finished mining {len(comment_map)} repos')


if __name__ == '__main__':
    main()
