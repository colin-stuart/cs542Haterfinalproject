import psycopg2
import numpy as np
import pandas

con1 = psycopg2.connect(dbname='xxx',
                        host='xxx',
                        port='xxx',
                        user='xxx',
                        password='xxx'
                        )  # first connection to Hater database
con1.set_client_encoding('UTF8')  # ensures that message text is properly encoded when saved to CSV file
cur1 = con1.cursor()
cur1.execute("SELECT M. userid, M.eventvalue \
            FROM mit.appmessages AS M \
            WHERE eventvalue IS NOT NULL\
            AND M.userid IN (SELECT S.userid FROM mit.scammers AS S)\
            LIMIT 1000\
            ")  # SQL command for spam userids and messages
spam = cur1.fetchall()
spam = np.array(spam)  # stores spam data in numpy array
spam = np.hstack(
    (np.ones((spam.shape[0], 1)), spam))  # inserts column of ones in front of spam data to label it as spam
cur1.close()  # close first connection to database

con2 = psycopg2.connect(dbname='hater',
                        host='hater-dw.czawprqluhe7.us-east-1.redshift.amazonaws.com',
                        port='5439',
                        user='mithater',
                        password='e7LAbuYhQYqSJswFaMAWQyWY4DFgC79E5K543RbJGrpfHKJIm9s1xePpGkv8fuC82v2Vd3tHlIKQijotiHkDJZQf79datE9kZ150Vov4V6K3fXmmsuN1js7er8fJAcTo'
                        )  # second connection to Hater database
con2.set_client_encoding('UTF8')  # ensures that message text is properly encoded when saved to CSV file
cur2 = con2.cursor()
cur2.execute("SELECT M. userid, M.eventvalue \
            FROM mit.appmessages AS M \
            WHERE eventvalue IS NOT NULL\
            AND M.userid NOT IN (SELECT S.userid FROM mit.scammers AS S)\
            LIMIT 1000\
            ")  # SQL command for non-spam userids and messages
ham = cur2.fetchall()
ham = np.array(ham)  # stores non-spam data in numpy array
ham = np.hstack(
    (np.zeros((ham.shape[0], 1)), ham))  # inserts column of ones in front of non-spam data to label it as non-spam
cur2.close()  # close second connection to database

labeled_data = np.concatenate((spam, ham), axis=0)  # joins spam & non-spam data into single array
df = pandas.DataFrame(data=labeled_data, columns=["is_spam", "userid", "message"])  # converts data to pandas dataframe
df.to_csv("labeled_messages.csv", sep=',', index=False)  # saves dataframe to CSV
