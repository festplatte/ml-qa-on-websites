import re
import jsonlines
import sys


def build_text_file(src_path, dest_path):
    input_path = dest_path + '.source'
    label_path = dest_path + '.target'
    with jsonlines.open(src_path) as data_json:
        input_file = open(input_path, 'w')
        label_file = open(label_path, 'w')

        input_data = ''
        label_data = ''

        for line in data_json.iter():
            qna_pair = "Frage: " + line['query'] + " "
            qna_pair = qna_pair + "Kontext: "
            for document in line['passages']:
                qna_pair = qna_pair + document['passage_text'] + " "
            qna_pair = re.sub(r"\s", " ", qna_pair)
            
            input_data += qna_pair + "\n"
            label_data += line['answers'][0] + "\n"
        input_file.write(input_data)
        label_file.write(label_data)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        build_text_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: prepare-seq2seq.py <input file> <desired outputname>")
