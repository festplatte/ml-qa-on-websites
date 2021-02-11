# Machine learning for question answering in the context of websites

This repository holds the code for my master thesis about building a question answering system in order to answer questions related to the content of websites. There are some scripts that can be used to fine-tune neural network based language models to the task of answer generation and code for webcrawlers that can index the websites used for this work.

## Abstract

This thesis investigates different approaches for answering questions based on the information of a web page and therefore considers the areas of website crawling, passage retrieval and generative question answering from the research area of natural language processing and information retrieval. To generate answers based on a given context, the pre-trained Transformer-based language models GPT-2 and T5 are fine-tuned on the MS Marco dataset. When evaluated on half of the dev set, the best GPT-2 model achieves a Rouge-L value of 0.625 and Bleu-1 of 0.577, and 0.495 and 0.390 for the best T5 model, respectively. Using a German GPT-2 model and a German translation of MS Marco created with T5, the same procedure yields values of 0.506 (Rouge-L) and 0.499 (Bleu-1) for GPT-2 and 0.437 (Rouge-L) and 0.375 (Bleu-1) for T5, respectively. The best trained language models are evaluated together with different web crawling methods and passage retrieval algorithms with example questions on three city web pages, where the best values for the questions and the web page of the Canadian city of Reddeer were achieved. This used a pipeline with HTML passages-based crawling, BM25 for passage retrieval and the trained T5 model for answer generation to achieve a Rouge-L value of 0.331 and a Bleu-1 value of 0.357. However, when comparing the generated answers to the human-written ones by human judges, the GPT-2 model outperformed the T5 model with 18.5 \% better and 22.7 \% equally good answers. The use of word embedding based passage retrieval algorithms mostly could not positively influence the response quality compared to BM25. Among the tested approaches for website crawling, splitting a website based on structural HTML tags and their semantic-based concatenation proved to be useful.

## Contents

- **citycrawler**: webcrawler built with scrapy to crawl the websites reddeer.ca, wuerzburg.de and nuernberg.de
- **evaluation**: scripts to evaluate the results of the experiments made
- **fine-tune-gpt2**: fine-tuning scripts for the GPT-2 model
- **fine-tune-T5**: fine-tuning scripts for the T5 model
- **human-evaluation-tool**: next.js based webapp to let humans rate generated answers
- **ms-marco-dataset**: scripts to prepare the data from the [MS Marco dataset](https://microsoft.github.io/msmarco/) to use it for fine-tuning or translation
- **translation**: scripts to translate the MS Marco dataset
