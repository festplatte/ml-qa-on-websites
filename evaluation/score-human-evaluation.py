from tqdm import tqdm
import jsonlines
import json
import sys

def score(input, mapping):
    with open(input, 'r') as input_file:
        inputs = json.load(input_file)

    with open(mapping, 'r') as mapping_file:
        mappings = json.load(mapping_file)

    scores = {
        mappings[0]["a1"]: 0,
        mappings[0]["a2"]: 0,
        "equal": 0
    }

    for i in tqdm(range(len(inputs))):
        cur_input = inputs[i]
        cur_mapping = mappings[i]

        if cur_input["a1"]["is_selected"] != "" and cur_input["a2"]["is_selected"] != "":
            scores["equal"] += 1
        elif cur_input["a1"]["is_selected"] != "":
            scores[cur_mapping["a1"]] += 1
        elif cur_input["a2"]["is_selected"] != "":
            scores[cur_mapping["a2"]] += 1

    print(scores)
    sum_ratings = 0
    for key in scores.keys():
        sum_ratings += scores[key]
    for key in scores.keys():
        scores[key] /= sum_ratings
    print(scores)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        score(sys.argv[1], sys.argv[2])
    else:
        print("Usage: score-human-evaluation.py <input file> <mapping file>")
