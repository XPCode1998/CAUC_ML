import csv

file1 = 'data/nlp_2.csv'
file2 = 'data/cv_2.csv'
file3 = 'data/paper_abstract.csv'

with open(file1, 'r', encoding='utf-8-sig') as f_in1, open(file2, 'r', encoding='utf-8-sig') as f_in2, open(file3, 'w',
                                                                                                    encoding='utf-8') as f_out:
    writer = csv.writer(f_out, delimiter=',')
    header_list = ['sentence', 'label']
    writer.writerow(header_list)
    for line in f_in1:
        if len(line.strip()) > 0:
            if line.startswith('"'):
                line = line[1:]
            if line.endswith('"'):
                line = line[:-1]
            abstract = line
            label = 0
            writer.writerow([abstract, label])
    for line in f_in2:
        if len(line.strip()) > 0:
            if line.startswith('"'):
                line = line[1:]
            if line.endswith('"'):
                line = line[:-1]
            abstract = line
            label = 1
            writer.writerow([abstract, label])

print("转换完成！")