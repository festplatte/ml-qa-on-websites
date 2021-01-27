from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import sys
from tqdm import tqdm
import jsonlines
import json


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        source_path = sys.argv[1]
        target_path = sys.argv[2]
        model_path = sys.argv[3]
        flags = sys.argv[4] if len(sys.argv) > 4 else ""
        cache_dir = '/data/.cache' if "c" in flags else None
    else:
        print("Usage: translate-ms-marco-dev.py <input file> <desired outputname> <model path> <flags>")

save_after = 1000

model = AutoModelWithLMHead.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

translator = pipeline("translation_en_to_de", model=model,
                      tokenizer=tokenizer, device=0)
def translate(text):
    max_length = len(text.split())*5
    return translator(text, max_length=400 if max_length < 400 else max_length)[0]['translation_text']

# MODEL_COMMAND = 'translate English to German: {}'
# def translate(text):
#     inputs = tokenizer.encode(MODEL_COMMAND.format(text), return_tensors="pt")
#     outputs = model.generate(inputs, max_length=len(
#         inputs[0])*2, early_stopping=True)
#     return tokenizer.decode(outputs[0])

def get_file_lines(file):
    try:
        with open(file, 'r') as file1:
            lines = file1.readlines()
            return len(lines)
    except IOError:
        return 0

source_content = ''
with open(source_path, 'r') as source_file:
    source_content = source_file.read()

source_file = source_content.splitlines()
num_target_lines = get_file_lines(target_path)
source_file = source_file[num_target_lines:]


with jsonlines.open(target_path, "a") as target_file:
    translated_lines = []
    for line in tqdm(source_file, initial=num_target_lines, total=len(source_file) + num_target_lines):
        line_json = json.loads(line)
        line_json['query'] = translate(line_json['query'])
        for passage_id, passage in enumerate(line_json['passages']):
            line_json['passages'][passage_id]['passage_text'] = translate(
                passage['passage_text'])
        for answer_id, answer in enumerate(line_json['answers']):
            line_json['answers'][answer_id] = translate(
                answer)
        print(line_json['query'])
        translated_lines.append(line_json)
        if len(translated_lines) >= save_after:
            print('saving')
            target_file.write_all(translated_lines)
            translated_lines = []
    if len(translated_lines) > 0:
        target_file.write_all(translated_lines)
