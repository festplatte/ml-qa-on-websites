from tqdm import tqdm
import json
import sys
import numpy as np

def check_text_length(file_path, filter_too_long = False):
    source_content = ''
    with open(file_path, 'r') as source_file:
        source_content = source_file.read()

    source_file = source_content.splitlines()

    text_lengths = []

    for i, line in enumerate(tqdm(source_file)):
        line_json = json.loads(line)
        answer = line_json['answers'][0]
        if filter_too_long and answer == "":
            continue
        text_lengths.append(len(answer))
        text_lengths.append(len(line_json['query']))

        for passage in line_json['passages']:
            text_lengths.append(len(passage['passage_text']))
    return text_lengths


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        text_lengths = check_text_length(sys.argv[1], "f" in sys.argv[2] if len(sys.argv) > 2 else False)
        print('avg text length:', np.average(text_lengths))
        print('stddr text length:', np.std(text_lengths))
        print('median text length:', np.median(text_lengths))
    else:
        print("Usage: score-ms-marco.py <input file>")
