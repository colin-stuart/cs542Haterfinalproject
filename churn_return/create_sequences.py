from cdb import CDB
from Data import Data
import numpy as np
import json

def get_days_over_100(file_name):
    #Function to Remove Data from days before day 100.
    #This function is required because the Message data
    #supplied to us starts 100 days after the swipe and question data.
    db = CDB()
    db.load(file_name)
    swipes = db.get_data()
    keys = swipes.keys()
    for key in keys:
        days = sorted(swipes[key].keys())
        for day in days:
            if int(day) < 100:
                swipes[key].pop(day)
    d = Data()
    d.set_data(swipes)
    db.set_data(d)
    db.save(file_name)

def delete_empty_dicts():
    db = CDB()
    db.load(file_name)
    swipes = db.get_data()
    keys = list(swipes.keys())
    for key in keys:
        if len(swipes[key].keys()) == 0:
            swipes.pop(key)
    d = Data()
    d.set_data(swipes)
    db.set_data(d)
    db.save(file_name)


def regularize_combine():
    f = open("hmm_data.txt","w")

    swipes_db = CDB()
    swipes_db.load("swipes.db")#Load swipe data

    questions_db = CDB()
    questions_db.load("questions.db")#load question data

    messages_db = CDB()
    messages_db.load("messages.db")#load message data

    swipes = swipes_db.get_data()#Data on the number of other users each user has swiped on each day
    questions = questions_db.get_data()#Data on the number of questions each user has answered on each day
    messages = messages_db.get_data()#Data on the number of messages each user has sent on each day

    keys = swipes.keys()
    for key in keys:#Where key is the userid of a user
        #Begin finding the earliest and latest days in the sequence for a specific user
        earliest_days = []#List of earliest days of activity. Will contain days of earliest swipe, earliest question answered, and earliest message sent
        latest_days = []#List of latest days of activity. Will contain days of latest swipe, latest question answered, and latest message sent
        key_swipes = swipes[key].keys()#A list of the days that the user swiped on at least 1 other user

        swipe_days = sorted(swipes[key].keys())#Sorted list of days that the user swiped on at least 1 other user
        earliest_days += [int(swipe_days[0])]#Extract earliest day that the user swiped on at least 1 other user and add it to list of earliest days
        latest_days += [int(swipe_days[-1])]#Extract latest day that the user swiped on at least 1 other user and add it to list of latest days

        key_questions = {}#Will be added to later if the user has answered any questions
        key_messages = {}#Will be added to later if the user has sent any messages

        if key in questions:#Check if the user has answered any questions
            key_questions = questions[key].keys()#List of days that the user has answered at least one question
            question_days = sorted(questions[key].keys())
            earliest_days += [int(question_days[0])]#Extract earliest day that the user answered at least 1 question and add it to list of earliest days
            latest_days += [int(question_days[-1])]#Extract latest day that the user answered at least 1 question and add it to list of latest days
        if key in messages:#Check if the user has sent any messages
            key_messages = messages[key].keys()#List of days that the user has sent at least one message
            messages_days = sorted(messages[key].keys())
            earliest_days += [int(messages_days[0])]
            latest_days += [int(messages_days[-1])]
        earliest = min(earliest_days)#Find the earliest day the user has done any of the above activities
        latest = max(latest_days)#Find the latest day the user has done any of the above activities

        mess = []#Sequence of number of messages sent from earliest day to latest day
        swip = []#Sequence of number of users swiped from earliest day to latest day
        quest = []#Sequence of number of questions answered from earliest day to latest day

        for i in range(earliest,latest+1):#Iterate through every day from earliest to latest
            i = str(i)
            if i in key_swipes:
                #Check if the user has swiped any other users on this day
                swip += [swipes[key][i]]#if they have, add the number to the sequence
            else:
                #If they haven't, add 0 to the sequence
                swip += [0]
            if i in key_questions:
                quest += [questions[key][i]]
            else:
                quest += [0]
            if i in key_messages:
                mess += [messages[key][i]]
            else:
                mess += [0]

        swip = np.array(swip)
        mess = np.array(mess)
        quest = np.array(quest)

        participation = swip + quest#Participation is defined as the number of other users swiped plus the number of questions answered

        #normalize participation
        part_mean = np.mean(participation)
        part_stdv = np.std(participation)
        participation = participation - part_mean
        if part_stdv != 0:
            participation = participation/part_stdv

        #Normalize messages
        mess_mean = np.mean(mess)
        mess_stdv = np.std(mess)
        mess = mess - mess_mean
        if mess_stdv != 0:
            mess = mess/mess_stdv

        #Activity is the mean of participation and messages sent
        activity = mess + participation
        activity = activity/2
        activity = list(activity)

        #Save the sequence to hmm_data.txt
        print(json.dumps(activity),file=f)
    f.close()

#Clean up data
get_days_over_100("questions.db")
get_days_over_100("swipes.db")
delete_empty_dicts("questions.db")
delete_empty_dicts("swipes.db")
#Convert data into desired sequences
regularize_combine()
