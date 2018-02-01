import csv

zero_count = 0
one_count = 0
with open("data/train.csv","rb") as csvfile:
# with codecs.open("data/train.csv", "r", encoding = "utf-8", errors = 'ignore') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        if(row['is_duplicate'] == "0"):
            zero_count += 1
        if(row['is_duplicate'] == "1"):
            one_count += 1

print zero_count
print one_count