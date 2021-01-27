from tqdm import tqdm
import jsonlines
import json
import sys
import random

random.seed(42)

def prepare(question_input, input1, input2, output):
    with open(question_input, 'r') as question_input_file:
        question_input_content = question_input_file.read()
        question_input_content = question_input_content.splitlines()

    with open(input1, 'r') as input1_file:
        input1_content = input1_file.read()
        input1_content = input1_content.splitlines()

    with open(input2, 'r') as input2_file:
        input2_content = input2_file.read()
        input2_content = input2_content.splitlines()

    input1 = input1.split("/").pop()
    input2 = input2.split("/").pop()
    contents = {
        input1: input1_content,
        input2: input2_content
    }
    inputs = [input1, input2]

    results = []
    mappings = []
    for i in tqdm(range(len(input1_content))):
        i1 = json.loads(question_input_content[i])

        random.shuffle(inputs)
        a1 = inputs[0]
        a2 = inputs[1]
        a1_json = json.loads(contents[a1][i])
        a2_json = json.loads(contents[a2][i])
        
        results.append({
            "id": i1["query_id"],
            "question": i1["query"],
            "a1": {
                "answer": a1_json["answers"][0],
                "is_selected": ""
            },
            "a2": {
                "answer": a2_json["answers"][0],
                "is_selected": ""
            }
        })
        mappings.append({
            "a1": a1,
            "a2": a2
        })

    with open(output, 'w') as output_file:
        json.dump(results, output_file)
    with open(output + ".mapping", 'w') as output_file:
        json.dump(mappings, output_file)

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        prepare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage: prepare-human-evaluation.py <question input file> <input file 1> <input file 2> <output file>")
