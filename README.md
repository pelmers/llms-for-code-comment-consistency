# Large Language Models for Code Comment Consistency

This repository hosts the replication package for the thesis project "Large Language Models for Code Comment Consistency" by Peter Elmers.

It is organized by research question. Each research question consists of self-contained Jupyter notebooks that reproduce the main results of the thesis.
That means the notebooks will download their data and dependencies on execution.
Change the parameters at the top of the file to adjust its experiment settings.

## RQ1: LLMs for Java Code Comment Consistency using data from Panthaplackel et al. (2020)
- Contains training notebooks for CodeBERT, Codegen, and replicated previous works (DeepJIT, BERT, Longformer).

## RQ2: Multi-lingual Code Comment Consistency with newly mined data
- Contains data collection scripts for Java, Python, JavaScript, and Go.
- Models are trained with the same notebooks as in **RQ1**. Make sure to update the data path.
- The data is hosted on Kaggle: https://www.kaggle.com/datasets/pelmers/multilingual-paired-code-and-comment-changes

## RQ3: Handpicked Benchmark Set of Code Comment Consistency
- Includes scripts that mine Github Pull Request Comments
- The `benchmarks` folder contains the final set of manually selected examples of inconsistent comments (25 per language).
- The raw pull request comment data is hosted on Kaggle: https://www.kaggle.com/datasets/pelmers/github-public-pull-request-comments

## RQ4: Social study, sending pull requests that fix inconsistent comments
- Consists of a notebook that will execute a comment consistency model on any open source repository.
