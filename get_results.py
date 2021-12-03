import os
import csv

dirs = [
    '.experiment',
]
exps = []
for dir in dirs:
    folders = os.listdir(dir)
    for folder in folders:
        relative_path = os.path.join(dir, folder)
        filenames = ['test_results.txt']
        for fn in filenames:
            result_file = os.path.join(relative_path, fn)
            if os.path.exists(result_file):
                with open(result_file) as f:
                    f1, prc, rc, acc = None, None, None, None
                    for line in f.readlines():
                        if line.startswith('f1'):
                            f1 = round(float(line.strip().split()[-1]), 4)
                        if line.startswith('precision'):
                            prc = round(float(line.strip().split()[-1]), 4)
                        if line.startswith('recall '):
                            rc = round(float(line.strip().split()[-1]), 4)
                        if line.startswith('accuracy'):
                            acc = round(float(line.strip().split()[-1]), 4)
                    exps.append([relative_path, fn, f1, prc, rc, acc])

with open('experimental_results.csv', mode='w') as f:
    f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    f.writerow(['experiment', 'filename', 'f1', 'precision', 'recall', 'accuracy'])
    f.writerows(exps)
print('finished writing file.')
