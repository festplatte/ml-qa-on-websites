from tqdm import tqdm
import json
import sys
import numpy as np

def check_answers_length(file_path, filter_too_long = False):
    source_content = ''
    with open(file_path, 'r') as source_file:
        source_content = source_file.read()

    source_file = source_content.splitlines()

    answer_lengths = []

    for i, line in enumerate(tqdm(source_file)):
        line_json = json.loads(line)
        answer = line_json['answers'][0]
        if filter_too_long and answer == "":
            continue
        answer_lengths.append(len(answer))
    return answer_lengths


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        answer_lengths = check_answers_length(sys.argv[1], "f" in sys.argv[2] if len(sys.argv) > 2 else False)
        print('avg answers length:', np.average(answer_lengths))
        print('stddr answers length:', np.std(answer_lengths))
        print('median answer length:', np.median(answer_lengths))
    else:
        print("Usage: score-ms-marco.py <input file>")
