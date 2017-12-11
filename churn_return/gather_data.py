from cdb import CDB
from Data import Data
import psycopg2
from date import Date


def get_days():
    #Function to assign a mapping of Date to number of days since December 6, 2016.
    #Returns Mapping.
    days = 0
    days_map = {}
    current_day = Date(12,6,2016)
    end = Date(9,25,2017)
    days_map[current_day] = days
    while True:
        current_day = current_day.copy()
        current_day.tomorrow()
        days += 1
        days_map[current_day] = days
        if current_day == end:
            break
    return days_map

def get_day(date,days):
    #Function that takes in a Date, and a mapping
    #And returns What the date maps to under the mapping.
    return days[date]



def gather_messages(days,con):
    cur = con.cursor()
    cur.execute("SELECT userid,eventdate FROM mit.appmessages WHERE eventdate < '2017-09-26 00:00:00' AND eventdate > '2016-12-06 00:00:00'")#Execute query
    data = cur.fetchone()
    messages = {}
    db = CDB()

    #Count number of messages sent by each user on each day, where a user is a dictioinary of days.
    #e.g. user "1" could look like: "1":{0:2, 5:4}
    #In this example, User 1 has sent 2 messages on day "0", and 4 messages on day "5"
    while data != None:
        user = data[0]
        date = data[1]
        day = get_day(Date(date.month,date.day,date.year),days)
        if user in messages:
            if day in messages[user]:
                messages[user][day] += 1
            else:
                messages[user][day] = 1
        else:
            messages[user] = {}
            messages[user][day] = 1
        data = cur.fetchone()
    d = Data()
    d.set_data(messages)
    db.set_data(d)
    db.save("messages.db")

def gather_questions(days,con):
    cur = con.cursor()
    cur.execute("SELECT userid,createddate FROM mit.appuserquestion")
    data = cur.fetchone()
    questions = {}
    db = CDB()

    #Count number of questioin answered by each user on each day, where a user is a dictioinary of days.
    #e.g. user "1" could look like: "1":{0:2, 5:4}
    #In this example, User 1 has answered 2 questions on day "0", and 4 questions on day "5"
    while data != None:
        user = data[0]
        date = data[1]
        day = get_day(Date(date.month,date.day,date.year),days)
        if user in questions:
            if day in questions[user]:
                questions[user][day] += 1
            else:
                questions[user][day] = 1
        else:
            questions[user] = {}
            questions[user][day] = 1
        data = cur.fetchone()
    d = Data()
    d.set_data(questions)
    db.set_data(d)
    db.save("questions.db")

def gather_swipes(days,con):
    cur = con.cursor()
    cur.execute("SELECT userid,liked,lastseen FROM mit.apprecommendation")# WHERE createddate < '2017-08-25 00:00:00' AND createddate > '2017-08-01 00:00:00';")
    data = cur.fetchone()
    swipes = {}

    #Count number of users swiped by each user on each day, where a user is a dictioinary of days.
    #e.g. user "1" could look like: "1":{0:2, 5:4}
    #In this example, User 1 has swiped on 2 other users on day "0", and 4 other users on day "5"
    while data != None:
        swipes = {}
        user = data[0]
        createddate = data[2]
        if swiped != 0:
            date = Date(createddate.month,createddate.day,createddate.year)
            day = get_day(date,days)
            if user in swipes:
                if day in swipes[user]:
                    swipes[user][day] += 1
                else:
                    swipes[user][day] = 1
            else:
                swipes[user] = {}
                swipes[user][day] = 1
        data = cur.fetchone()

    d = Data()
    d.set_data(swipes)
    db = CDB()
    db.set_data(d)
    db.save("swipes.db")

con=psycopg2.connect(dbname= 'xxx', host='xxx', port= 'xxx', user= 'xxx', password= 'xxx')#Connect to hater database
days = get_days()
gather_messages(days,con)
gather_questions(days,con)
gather_swipes(days,con)
