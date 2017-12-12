import psycopg2
import numpy as np

# Connecting to Hater's DB server

# xxx in order to protect Hater's database from public

con=psycopg2.connect(dbname= 'xxx', host='xxx',
port= 'xxx', user= 'xxx', password= 'xxx')


###### FILE 1 #######

# Using mit.appuser

# Getting all user data - ID, age, minageinterest, maxageinterest, sex, genderpref, lat, long, dist, attractiveness, picture

cur = con.cursor()
file = open("user_data.csv", "w")
cur.execute("SELECT M.id, M.age, M.minageinterest, M.maxageinterest, M.sex, M.interest, M.latitude, M.longitude, M.distance, M.attractiveness\
            FROM mit.appuser as M ")

# Writing to the file called user_data.txt

for row in cur:
    print(list(row), file=file)
file.close()

###### FILE 2 #########

# Finding true matches - Using 'Dater - Recommendation answered - v2'

# Using mit.event

cur = con.cursor()
file = open("true_matches_nf.txt", "w")

cur.execute("SELECT DISTINCT(MA.property2_value), MA.userid, MA.property1_value \
            FROM mit.event as MA \
            WHERE eventname LIKE 'Dater - Recommendation answered - v2'\
                  AND eventdate >= '2017-08-20' AND MA.property2_value != '-1'\
            ORDER BY MA.property2_value ASC LIMIT 407438")

# Writing true matches to a text file called true_matches

# First field: userid. Second field: otheruserid. Third field: Liked or disliked.

for row in cur:
    print(list(row), file=file)
file.close()

###### FILE 3 #######

# Getting the userid, topics, and their feeling towards it

# Using mit.appuserquestion

cur = con.cursor()
file = open("user_question.csv", "w")

cur.execute("SELECT AQ.userid, AQ.questionid, AQ.answer FROM mit.appuserquestion as AQ")

for row in cur:
    print(list(row), file=file)
file.close()

####### FILE 4 ######

# For getting the true match data

# Using mit.event

cur = con.cursor()
file = open("user_true_match_data_nf.txt", "w")    # _nf denotes 'not final' - final version available from datafixer.py

cur.execute("SELECT M.id, M.age, M.sex, M.latitude, M.longitude, M.attractiveness\
              FROM mit.appuser as M \
              WHERE M.id IN (SELECT DISTINCT(MA.property2_value)\
                                  FROM mit.event as MA \
                                  WHERE eventname LIKE 'Dater - Recommendation answered - v2'\
                                        AND eventdate >= '2017-08-20'\
                                  ORDER BY MA.property2_value ASC) ORDER BY M.id ASC")

for row in cur:
    print(list(row), file=file)
file.close()

########## FILE 5 #########

# Gets users pictures to calculate extroversion score

cur = con.cursor()
file = open("user_pictures.txt", "w")
cur.execute("SELECT M.pictures2\
            FROM mit.appuser as M ORDER BY M.id ASC")

# Writing to the file called user_pictures.txt

for row in cur:
    print(list(row), file=file)
file.close()




