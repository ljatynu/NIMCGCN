import csv
from ast import literal_eval as make_list

a = open("F:/GCN-IMC-MD/code/result.txt", "r")
strstr = a.readlines()
str1 = make_list(strstr[0])
str2 = make_list(strstr[1])

out1 = open('fpr_model.csv', 'w', newline='')
out2 = open('tpr_model.csv', 'w', newline='')
csv_write1 = csv.writer(out1, dialect="excel")
csv_write2 = csv.writer(out2, dialect="excel")
for i in str1:
    csv_write1.writerow([str(i)])

for i in str2:
    csv_write2.writerow([str(i)])


out1.close()
out2.close()