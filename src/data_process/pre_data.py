import re
import pandas as pd
import os
from paths import Paths


def csv2txt(version_dir):
    df = pd.read_csv(version_dir / "matrix.csv", index_col=0)
    df.to_csv(version_dir / "matrix.csv", index=False)
    tag = df['error']
    df = df.drop(['error'], axis=1, inplace=False)
    df.replace(",", " ")
    df.to_csv(version_dir / "matrix.txt", index=False, sep=' ', header=0)
    tag.to_csv(version_dir / "error.txt", index=False, header=0)


def pre_raw(version_dir):
    line_list = []
    file_path = version_dir / "spectra"
    with open(file_path, 'r') as f:  #
        lines = f.readlines()
        for line in lines:
            left, right = line.split("#")
            line_list.append(re.findall('\d+', right)[0])
    line_list.append('error')

    data = []
    path = version_dir / "matrix"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            line = line.split()
            if line[-1] == '+':
                line[-1] = '0'
            if line[-1] == '-':
                line[-1] = '1'
            data.append(line)
    test = pd.DataFrame(columns=line_list, data=data)
    test.to_csv(version_dir / "matrix.csv", index=0)


def main():
    program = "Chart"
    program_data_path = Paths.get_d4j_program_data_dir(program)
    versions = os.listdir(program_data_path)
    for version in versions:
        version_dir = Paths.get_d4j_version_dir(program, version)
        csv2txt(version_dir)
        pre_raw(version_dir)


if __name__ == '__main__':
    main()
