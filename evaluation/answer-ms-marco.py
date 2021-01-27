from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import jsonlines
import json
import re
import sys
import torch
import numpy as np

from answerers import find_answerer

torch.manual_seed(42)
np.random.seed(42)
# torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'

cache_dir = '/data/.cache'
save_after = 1000

# if torch.cuda.is_available():
#     model.cuda()

def map_passages(passages):
    result = []
    for passage in passages:
        result.append(passage['passage_text'])
    return result

def get_file_lines(file):
    try:
        with open(file, 'r') as file1:
            lines = file1.readlines()
            return len(lines)
    except IOError:
        return 0

def generate_answers(source_path, target_path, answerer):
    source_content = ''
    with open(source_path, 'r') as source_file:
        source_content = source_file.read()

    source_file = source_content.splitlines()
    num_target_lines = get_file_lines(target_path)
    source_file = source_file[num_target_lines:]

    with jsonlines.open(target_path, "a") as target_file:
        processed_lines = []
        for line in tqdm(source_file, initial=num_target_lines, total=len(source_file) + num_target_lines):
            line_json = json.loads(line)
            line_json['generated_answer'] = answerer.generate_answer(line_json['query'], map_passages(line_json['passages']))

            # save results
            processed_lines.append({
                "query_id": line_json["query_id"],
                "answers": [line_json["generated_answer"]],
                "reference_answers": line_json["answers"]
            })

            # print results
            print('query: ' + line_json['query'])
            print('generated answer: ' + line_json['generated_answer'])

            # save to file
            if len(processed_lines) >= save_after:
                print('saving')
                target_file.write_all(processed_lines)
                processed_lines = []
        if len(processed_lines) > 0:
            target_file.write_all(processed_lines)

def main(args):
    source_path = args[1]
    target_path = args[2]
    model_path = args[3]
    flags = args[4] if len(args) > 4 else ""
    AnswererClass = find_answerer(model_path)
    answerer = AnswererClass(model_path, cache_dir=cache_dir if "c" in flags else None)

    generate_answers(source_path, target_path, answerer)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        main(sys.argv)
    else:
        print("Usage: answer-ms-marco.py <input file> <desired outputname> <model path> <flags>")
