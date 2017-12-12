# A file to fix the data retrieved from Hater's database - writes the formatted data onto a new file

import numpy as np
import json

finished = []
file_name = "user_question.csv"
f = open(file_name,"r")
for line in f:
    splitted = line.split(",")
    splitted[0] = splitted[0][1:]
    splitted[0] = int(splitted[0])
    splitted[1] = int(splitted[1])
    splitted[2] = splitted[2][:-2]
    splitted[2] = int(splitted[2])

#     USER QUESTION HAS ONLY 3 fields, whereas other files have more than 3, hence these extra commented lines

#     splitted[2] = int(splitted[2])
#     splitted[3] = splitted[3][1:]
#     splitted[3] = float(splitted[3])
#     splitted[4] = splitted[4][1:]
#     splitted[4] = float(splitted[4])
#     splitted[5] = splitted[5][1:]
#     splitted[5] = float(splitted[5][:-2])


    finished += [splitted]

f = open("user_question_f.csv","w")
print(json.dumps(finished),file=f)
f.close()
