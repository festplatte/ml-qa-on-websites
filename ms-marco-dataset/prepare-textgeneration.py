import re
import jsonlines
import sys


def build_text_file(src_path, dest_path):
    with jsonlines.open(src_path) as data_json:
        f = open(dest_path, 'w')
        data = ''
        for line in data_json.iter():
            qna_pair = ""
            qna_pair = qna_pair + "<BOS><SOQ>" + line['query'] + "\n"
            qna_pair = qna_pair + "<SOC>"
            for document in line['passages']:
                qna_pair = qna_pair + \
                    re.sub(r"\s", " ", document['passage_text']) + "\n"
            qna_pair = qna_pair + "<SOA>" + line['answers'][0] + "<EOS>\n"

            # qna_pair = re.sub(r"\s", " ", qna_pair)
            data += qna_pair
        f.write(data)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        build_text_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: prepare-textgeneration.py <input file> <desired outputname>")
