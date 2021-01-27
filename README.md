# Machine learning for question answering in the context of websites

This repository holds the code for my master thesis about building a question answering system in order to answer questions related to the content of websites. There are some scripts that can be used to fine-tune neural network based language models to the task of answer generation and code for webcrawlers that can index the websites used for this work.

## Abstract

## Contents

- **citycrawler**: webcrawler built with scrapy to crawl the websites reddeer.ca, wuerzburg.de and nuernberg.de
- **evaluation**: scripts to evaluate the results of the experiments made
- **fine-tune-gpt2**: fine-tuning scripts for the GPT-2 model
- **fine-tune-T5**: fine-tuning scripts for the T5 model
- **human-evaluation-tool**: next.js based webapp to let humans rate generated answers
- **ms-marco-dataset**: scripts to prepare the data from the [MS Marco dataset](https://microsoft.github.io/msmarco/) to use it for fine-tuning or translation
- **translation**: scripts to translate the MS Marco dataset
