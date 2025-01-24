import os

programs = ['clac', 'grep', 'PrimeCount', 'quich', 'sed', 'SeqMap', 'ShortestPath']
files = ['bugline.csv', 'matrix.csv', 'matrix', 'spectra', 'matrix.txt', 'error.txt']

def delete():
    path = './data/d4j/data'
    for program in programs:
        program_path = os.path.join(path, program)
        for version in os.listdir(program_path):
            version_path = os.path.join(program_path, version, 'gzoltars', program, version)
            for file in os.listdir(version_path):
                if file in files:
                    continue
                file_path = os.path.join(version_path, file)
                os.remove(file_path)

if __name__ == '__main__':
    delete()