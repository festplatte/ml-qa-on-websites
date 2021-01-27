import re
import jsonlines
import sys

MAX_CHARS = 1000000
MAX_SKIP_ENTRIES = 100


def build_text_file(src_path, dest_path):
    with jsonlines.open(src_path) as data_json:
        f = open(dest_path, 'w')
        data = ''
        count_entries = 0
        count_skips = 0
        for i, line in enumerate(data_json):
            qna_pair = line['query'] + "\n"
            for document in line['passages']:
                qna_pair += re.sub(r"\s", " ", document['passage_text']) + "\n"
            qna_pair += line['answers'][0] + "\n"

            # qna_pair = re.sub(r"\s", " ", qna_pair)
            if len(data) + len(qna_pair) > MAX_CHARS:
                if count_skips == 0:
                    print('last non-skipped entry:', i)
                count_skips += 1
                if count_skips > MAX_SKIP_ENTRIES:
                    break
            else:
                if count_skips > 0:
                    print('processed skipped entry:', i)
                data += qna_pair
                count_entries += 1
        f.write(data)
        print('processed entries:', count_entries, '- skiped entries:',
              count_skips, '- total chars:', len(data))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        build_text_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: prepare-deepl.py <input file> <desired outputname>")
