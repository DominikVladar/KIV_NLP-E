# NLP-E 2025 - Natural Language Processing

> **📚 University Coursework**  
> This repository contains my solutions for the NLP-E (Natural Language Processing in English) course assignments at the University of West Bohemia (ZČU), Faculty of Applied Sciences, Department of Computer Science and Engineering.

## Overview

The course covers fundamental and advanced NLP techniques through practical assignments:

| Assignment | Topic      | Description                                 |
| ---------- | ---------- | ------------------------------------------- |
| CV01       | MNIST & MC | Neural network basics, MNIST classification |
| CV02       | STS        | Semantic Text Similarity                    |
| CV03       | Sentiment  | Sentiment Classification with CNN           |
| CV04       | NER        | Named Entity Recognition with RNN/LSTM      |
| CV05       | LLM        | Large Language Models (OpenRouter API)      |

Each folder (`cv01/` - `cv05/`) contains:

- Assignment implementation
- Individual README with task description
- Unit tests
- Results and analysis

## Results

Summary of my results can be found in the [results/](results/) folder.

---

## Original Course Instructions

_The following section contains the original course setup instructions._

## How to Start

1. Each of you should see your own repository named `anlp-<year>_<surname>_<name>` where you are going to develop your solutions.
2. Use `git clone <repo_url>` to clone your own repo.
3. Each of you should also see `ANLP2025 Main` repository, where we are going to release the assignment templates.
4. Add another remote upstream using `git remote add assignments git@gitlab.kiv.zcu.cz:anlp/ANLP2025-main.git`.
5. Run `git pull assignments main` before starting each assignment to download the latest templates.
6. Use `git push` to push the template to your own repository.
7. Follow the instructions under `cv<xy>` folder to work on the assignment.

## General Rules and Instructions for the Assignments

### Wandb

To track your experiments, you are going to use the `wandb` library. When assessing your assignments we will investigate the wandb logs of your experiments under your own project. To start working with the wandb, do the following steps:

1. Create a new account at [Weights&Biases](https://wandb.ai/).
2. From the root of your repository run the following from commandline: `pip3 install wandb`
3. From the root of your repository run the following from commandline: `wandb login`
4. When prompted, use your own API key from wandb.ai settings.
5. Change `WANDB_PROJECT` in `wandb_config.py` by filling your name and username into template `anlp-<year>_<surname>_<name>`

### Unittests

There's a set of unittests prepared under `cv<xy>/test` folder for each of the assignments. Those unittests, are there for you to make sure that your solution is working as expected. Solutions with failing unittests are not going to be assessed at all. In case you have any problems, there's always one support practical lesson designated for consulting these issues with you.

To run the unittests, please follow the subsequent instructions:

1. Set python home to root of the repository: `export PYTHONPATH=<root of your repository>`
2. Go to the test folder, for example: `cd cv01/test`
3. Run the following command: `python3 -m unittest discover`

Note: you can also run the tests from your IDE such as PyCharm.

### Submission

There are no explicit steps required to submit your solution. The only important thing is that your solution is in the `main` branch of your repository till the deadline (commit date & time must be before the deadline).
