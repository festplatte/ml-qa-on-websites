import re
import jsonlines
import sys


def build_text_file(src1_path, src2_path, dest_path):
    with open(src1_path) as src1:
        with open(src2_path) as src2:
            with open(dest_path, 'w') as dest:
                src1_lines = src1.readlines()
                src2_lines = src2.readlines()

                for i in range(0, len(src1_lines)):
                    dest.write(src1_lines[i])
                    dest.write(src2_lines[i])


if __name__ == '__main__':
    if len(sys.argv) == 4:
        build_text_file(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: prepare-mixed-dataset.py <input file> <input file> <desired outputname>")
