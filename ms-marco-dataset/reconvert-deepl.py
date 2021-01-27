import re
import jsonlines
import sys


def build_text_file(src_dataset_path, src_translation_path, dest_path):
    with jsonlines.open(src_dataset_path, 'r') as src_file:
        with open(src_translation_path, 'r') as translation_file:
            with jsonlines.open(dest_path, 'w') as dest_file:
                translation_lines = translation_file.readlines()

                translation_counter = 0
                src_counter = 0
                for data in src_file:
                # while translation_counter < len(translation_lines):
                #     data = src_file[src_counter]
                    data['query'] = translation_lines[translation_counter].strip()
                    translation_counter += 1

                    for i, _ in enumerate(data['passages']):
                        data['passages'][i]['passage_text'] = translation_lines[translation_counter].strip()
                        translation_counter += 1
                    
                    data['answers'][0] = translation_lines[translation_counter].strip()
                    translation_counter += 1
                    src_counter += 1
                    dest_file.write(data)
                    if translation_counter >= len(translation_lines):
                        break

                # data = ''
                # count_entries = 0
                # count_skips = 0
                # for i, line in enumerate(data_json):
                #     qna_pair = line['query'] + "\n"
                #     for document in line['passages']:
                #         qna_pair += re.sub(r"\s", " ",
                #                         document['passage_text']) + "\n"
                #     qna_pair += line['answers'][0] + "\n"

                #     # qna_pair = re.sub(r"\s", " ", qna_pair)
                #     if len(data) + len(qna_pair) > MAX_CHARS:
                #         if count_skips == 0:
                #             print('last non-skipped entry:', i)
                #         count_skips += 1
                #         if count_skips > MAX_SKIP_ENTRIES:
                #             break
                #     else:
                #         if count_skips > 0:
                #             print('processed skipped entry:', i)
                #         data += qna_pair
                #         count_entries += 1
                # f.write(data)
                # print('processed entries:', count_entries, '- skiped entries:',
                #     count_skips, '- total chars:', len(data))


if __name__ == '__main__':
    if len(sys.argv) == 4:
        build_text_file(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: reconvert-deepl.py <src dataset path> <src translation path> <target path>")
