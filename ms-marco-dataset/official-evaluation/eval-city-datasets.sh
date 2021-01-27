#!/bin/bash

./run.sh ../../data/city-datasets/reddeer-faq.jsonl ../../data/city-datasets/reddeer-faq_answered_gpt2_$1.jsonl
./run.sh ../../data/city-datasets/reddeer-faq.jsonl ../../data/city-datasets/reddeer-faq_answered_t5_$1.jsonl
./run.sh ../../data/city-datasets/wuerzbot.jsonl ../../data/city-datasets/wuerzbot_answered_gpt2_$1.jsonl
./run.sh ../../data/city-datasets/wuerzbot.jsonl ../../data/city-datasets/wuerzbot_answered_t5_$1.jsonl
./run.sh ../../data/city-datasets/nbg-faq.jsonl ../../data/city-datasets/nbg-faq_answered_gpt2_$1.jsonl
./run.sh ../../data/city-datasets/nbg-faq.jsonl ../../data/city-datasets/nbg-faq_answered_t5_$1.jsonl
