from tqdm import tqdm
import jsonlines
import json
import sys
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def rouge_l(source, target):
    return scorer.score(source, target)['rougeL'].fmeasure

def bleu(source, target):
    return sentence_bleu([source.split()], target.split(), weights=(0.5,0.5))

def score(input1_path, input2_path, debug = False, filter_too_long = False):
    input1_content = ''
    with open(input1_path, 'r') as input1_file:
        input1_content = input1_file.read()
    input1_file = input1_content.splitlines()

    input2_content = ''
    with open(input2_path, 'r') as input2_file:
        input2_content = input2_file.read()
    input2_file = input2_content.splitlines()

    rouge_l_scores = []
    bleu_scores = []

    for i in tqdm(range(len(input1_file))):
        json1 = json.loads(input1_file[i])
        json2 = json.loads(input2_file[i])

        rouge_l_scores.append(rouge_l(json1["query"], json2["query"]))
        bleu_scores.append(bleu(json1["query"], json2["query"]))

        rouge_l_scores.append(rouge_l(json1['answers'][0], json2['answers'][0]))
        bleu_scores.append(bleu(json1['answers'][0], json2['answers'][0]))

        for j in range(len(json1["passages"])):
            p1 = json1["passages"][j]["passage_text"]
            p2 = json2["passages"][j]["passage_text"]
            rouge_l_scores.append(rouge_l(p1, p2))
            bleu_scores.append(bleu(p1, p2))
    return np.average(rouge_l_scores), np.average(bleu_scores)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        avg_rouge_l, avg_bleu = score(sys.argv[1], sys.argv[2], "d" in sys.argv[3] if len(sys.argv) > 3 else False, "f" in sys.argv[3] if len(sys.argv) > 3 else False)
        print('final avg rougeL:', avg_rouge_l)
        print('final avg bleu:', avg_bleu)
    else:
        print("Usage: score-ms-marco.py <input1 file> <input2 file>")
