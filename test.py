import csv

with open('data/test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(reader):
        if i >= 100:
            break
        print(', '.join(row))
