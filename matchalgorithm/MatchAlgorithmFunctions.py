# Import modules
import numpy as np
import psycopg2


# con=psycopg2.connect(dbname= 'xxx', host='xxx',
# port= 'xxx', user= 'xxx', password= 'xxx')

# This file contains important functions for the match algorithm for the company Hater

def filter_matches(user_data):
    """Given user's information and preferences (such as min_Age, max_Age, etc.) will locate
    the ids of users who meet those criteria and pull them from the Hater Database."""

    # user_data: [id, age, min_Age, max_Age, sex, interest, lat, long, max_dist]

    user_age = user_data[0][1]
    min_age = user_data[0][2]
    max_age = user_data[0][3]
    user_sex = user_data[0][4]
    interest = user_data[0][5]
    latitude = user_data[0][6]
    longitude = user_data[0][7]
    max_dist = user_data[0][8]
    min_lat = latitude - max_dist
    max_lat = latitude + max_dist
    min_long = longitude - max_dist
    max_long = longitude + max_dist

    print('Initial filter \n')
    cur = con.cursor()
    select_stm = "SELECT M.id, M.age, M.pictures2, M.sex, M.interest, M.latitude, M.longitude, M.distance, M.attractiveness \
                FROM mit.appuser as M \
                WHERE M.id IN (SELECT M.id \
                                  FROM mit.appuser as M \
                                  WHERE M.age BETWEEN %(min_age)s AND %(max_age)s \
                                  AND %(user_age)s BETWEEN M.minageinterest AND M.maxageinterest \
                                  AND M.sex = %(interest)s OR %(interest)s = 2 \
                                  AND M.latitude BETWEEN %(min_lat)s AND %(max_lat)s \
                                  AND M.longitude BETWEEN %(min_long)s AND %(max_long)s \
                                   ) \
                 ORDER BY M.id ASC"
    values = {'min_age': min_age, 'max_age': max_age, 'user_age': user_age, 'interest': interest, 'user_sex': user_sex,
              'min_lat': min_lat, 'max_lat': max_lat, 'min_long': min_long, 'max_long': max_long, 'lat': latitude,
              'long': longitude}
    cur.execute(select_stm, values)
    prelim_matches = np.array(cur.fetchall())
    print('Initial potential matches: ', prelim_matches.shape[0])
    print(prelim_matches[0:10, 0])
    print('CHECKING DISTANCE')
    # Checks to make sure that the user and potential matches are within the max_dist requirement
    poss_matches = calculate_distance(latitude, longitude, max_dist, prelim_matches)
    print('CHECKING SEX INTEREST')
    # Checks that the user's sex matches the interest of the possible matches
    matches = check_sex_interest(user_sex, poss_matches)
    print('Total matches after filters: ', matches.shape[0])
    return matches


def gender_pref(prev_matches_sex):
    """If a user prefers both sexes (a 2 in the interest column), this will determine
    if the user prefers one gender over the other."""

    num_female = len(np.where(prev_matches_sex == 1)[0])
    num_male = len(np.where(prev_matches_sex == 0)[0])
    pref_female = num_female / (num_male + num_female)
    pref_male = 1 - pref_female
    return pref_female, pref_male


def extroversion_score(user_pictures):
    """Determines the level of extroversion of a user based on the number of pictures
    the user has. Input: user_pictures is a list of number of pictures for each user. Output is a list of extroversion scores"""

    # maximum number of photos: 5
    score = np.zeros(len(user_pictures))
    for i in user_pictures:
        score[i] = user_pictures[i] / 5
    return score


def calculate_distance(latitude, longitude, max_dist, matches):
    """Calculates the Euclidean Distance between two users"""
    # matches: [id, age, pict, sex, interest, lat, long, dist]

    delta_lat = np.subtract(latitude, matches[:, 5].astype(float))
    delta_long = np.subtract(longitude, matches[:, 6].astype(float))
    distance = np.sqrt(np.power(delta_lat, 2) + np.power(delta_long, 2))

    match_dist = matches[np.where((distance <= max_dist) & (distance <= matches[:, 7].astype(float))), :]
    return match_dist[0]


def check_sex_interest(user_sex, matches):
    """Determines whether users and fit each others sex interests"""
    # matches: [id, age, pict, sex, interest, lat, long, dist]

    match_interest = matches[np.where((user_sex == matches[:, 4].astype(int)) | (matches[:, 4].astype(int) == 2)), :]
    return match_interest[0]


def age_difference_penalty(user_age, matches):
    """Can be modified instead to determine if someone who is under 21 is trying to match
    with someone over 21 and penalizes that."""
    # Returns a 0 or 1 depending on whether or not there is a penalty

    # This function is not used anymore as it reduces model accuracy and matches

    penalty = np.zeros((matches.shape[0]))
    if user_age < 21:
        penalty[np.where(matches >= 21)] = 1
    else:
        penalty[np.where(matches < 21)] = 1
    return penalty
