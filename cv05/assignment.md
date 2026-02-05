# KIV / ANLP Exercise 05

**Deadline**: 17. 12. 2025 23:59:59

**Maximum points:** 20 + *5*

**Contact:** `hejmanj@kiv.zcu.cz` - in case of issues or questions about the assignment.

## Assignment

The goal of this exercise is to complete the three NLP tasks that you already know using LLMs.

**Constraints:**
- Only prompted *existing* LLM models that are found on OpenRouter.
- No test-set leakage on your part.

| Task                     | Dataset            | Abbreviation |
|--------------------------|--------------------|--------------|
| Sentiment Classification | ČSFD               | `csfd`       |
| Semantic Text Similarity | Czech News Dataset | `sts`        |
| Named Entity Recognition | CNEC               | `ner`        |

### Details
You are only allowed to use pre-trained LLMs with no additional training and prompting only.

LLMs that are available via [OpenRouter](https://openrouter.ai/) are allowed. You can still run the models locally or on MetaCentrum, as long as they are listed.

> In practice, fine-tuning LLMs is just as useful and popular as with smaller models, but the approach is fairly similar to what you saw in the previous exercise. 

Any techniques you use must not allow for any *potential* test-set leakage. In case you choose an agentic approach, watch out for memory layers and unrestricted file or web access.

In the case of test-set contamination of the foundational models, that is not your fault and you are free to find and abuse such models. I expect it to be relatively unlikely given our choice of datasets.

In order to achieve better results you can improve your [prompting](https://www.promptingguide.ai/), create [workflows](https://www.anthropic.com/engineering/building-effective-agents) or adjust your system in any other way. Structured outputs, tools, multi-agent systems, etc. are all allowed.

## In This Repository

```bash
├── assignment.md    # this file
├── README.md        # for your documentation
├── requirements.txt
├── data             # all required data, test sets are subsampled
│   ├── anlp01-sts-free-test-llm.tsv
│   ├── anlp01-sts-free-train.tsv   
│   ├── csfd-test-llm.tsv
│   ├── csfd-train.tsv
│   ├── ner-dev-llm.txt
│   └── ner-train.txt
├── src
│   ├── evaluator.py            # evaluator script
│   └── openrouter_key_info.py  # credits status utility for API
└── submissions     # your submissions go here
```

## API Keys

OpenRouter is a service that allows you to use a variety of models easily. API keys with $2 of OpenRouter credits are available on CourseWare.

The script `scr/openrouter_key_info.py` gives you usage information about your key.

In case you have plans for more extensive experiments, some options are:
1. Self-host the model using MetaCentrum or your own computer. [vLLM](https://docs.vllm.ai/en/latest/) and [llama.cpp](https://github.com/ggml-org/llama.cpp) are popular options.
2. Choose free models and deal with the rate limits.
    - Shared OpenRouter account is limited to 1000 free requests daily.
    - [Curated list](https://github.com/cheahjs/free-llm-api-resources#nvidia-nim) of other options - OpenAI compatible ones so that your code can swap providers.
3. You may ask for extra credits if you have plans for a particularly interesting experiment.

## Evaluation

The `src/evaluator.py` script gives you your official scores.
- When ran without arguments, it will scan the entire `submissions` directory and find best scores.
- You are not allowed to modify this script or the dataset.
- For more details, read the header of the evaluator script or its help.

To make experimentation more accessible, all datasets' test and dev set have been dramatically subsampled.

Please, include and keep as many results in the `submissions` directory, even if they are not your best results.

## Grading

**For completing each of the 3 tasks, 4 points are awarded:**
- **2 points** for a generally functional implementation
- **2 points** for beating the threshold score for each task:
    - `csfd-accuracy` > 0.7
    - `sts-mse`       < 2.85
    - `ner-f1-macro`  > 0.2

**The remaining 8 points are for the readme/documentation file:**
- **2 points** for description of your best system for each task
- **2 points** for result and task analysis
- **2 points** for having a visual of any sort, must be relevant and useful
- **1 point** for documenting the models you tried and how they performed
- **1 point** for explaining the sampling parameters, prompts, etc. used

**Bonus points:** Up to 5 bonus points per student may be awarded.  
Points over this boundary will be evenly redistributed to the next highest performer for each criterion.

- **2 points** for best score on each task
    - Reference metrics same as thresholds above.
- **3 points** for best readme
    - I am looking for interesting insights and quality analysis.
