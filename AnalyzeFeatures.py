# Import modules

import numpy as np
import scipy.stats as st
import pandas as pd

data = pd.read_csv("sample.csv")
print("Imported data")
gd = data.groupby('userid')
print("Reformatted data")


def euclidean_similarity(person1, person2):
    """Computes a similarity score using the Euclidean distance, which is between 0 and 1"""
    if person1 in gd.groups.keys():
        user_data_1 = gd.get_group(person1)
    else:
        return 0
    if person2 in gd.groups.keys():
        user_data_2 = gd.get_group(person2)
    else:
        return 0
    pivot_1 = user_data_1.pivot(index='userid', columns='topic_id', values='rating')
    pivot_2 = user_data_2.pivot(index='userid', columns='topic_id', values='rating')
    topic_1 = pivot_1.columns
    topic_2 = pivot_2.columns
    common_topics = np.intersect1d(topic_1, topic_2)
    if len(common_topics) != 0:
        rating_1 = np.zeros(len(common_topics))
        rating_2 = np.zeros(len(common_topics))
        idx = 0
        for ii in common_topics:
            rating_1[idx] = pivot_1[ii].iloc[0]
            rating_2[idx] = pivot_2[ii].iloc[0]
            idx += 1
        score = np.sqrt(np.sum(np.power(rating_1 - rating_2, 2)))
        return 1 / (1 + score)
    return 0


def pearson_correlation(person1, person2):
    """Computes a similarity score using the Pearson Coefficient, which is between -1 and 1"""
    user_data_1 = gd.get_group(person1)
    user_data_2 = gd.get_group(person2)
    pivot_1 = user_data_1.pivot(index='userid', columns='topic_id', values='rating')
    pivot_2 = user_data_2.pivot(index='userid', columns='topic_id', values='rating')
    topic_1 = pivot_1.columns
    topic_2 = pivot_2.columns
    common_topics = np.intersect1d(topic_1, topic_2)
    if len(common_topics) > 1:
        rating_1 = np.zeros(len(common_topics))
        rating_2 = np.zeros(len(common_topics))
        idx = 0
        for ii in common_topics:
            rating_1[idx] = pivot_1[ii].iloc[0]
            rating_2[idx] = pivot_2[ii].iloc[0]
            idx += 1
        score = st.pearsonr(rating_1, rating_2)[0]
        return abs(score)
    return 0


def spearman_similarity(person1, person2):
    """Computes a similarity score using the Spearman Rank Coefficient, which is between -1 and 1"""
    user_data_1 = gd.get_group(person1)
    user_data_2 = gd.get_group(person2)
    pivot_1 = user_data_1.pivot(index='userid', columns='topic_id', values='rating')
    pivot_2 = user_data_2.pivot(index='userid', columns='topic_id', values='rating')
    topic_1 = pivot_1.columns
    topic_2 = pivot_2.columns
    common_topics = np.intersect1d(topic_1, topic_2)
    if len(common_topics) != 0:
        rating_1 = np.zeros(len(common_topics))
        rating_2 = np.zeros(len(common_topics))
        idx = 0
        for ii in common_topics:
            rating_1[idx] = pivot_1[ii].iloc[0]
            rating_2[idx] = pivot_2[ii].iloc[0]
            idx += 1
        score = st.spearmanr(rating_1, rating_2)[0]
        return abs(score)
    return 0


def generate_similarity_scores(similarity_function):
    """Computes a similarity score for all users against all other users."""
    unique_ids = data['userid'].unique()
    val = (len(unique_ids) * (len(unique_ids) - 1)) / 2
    similarity_matrix = np.zeros([int(val), 3])
    idx = 0
    for p1_idx in np.arange(len(unique_ids)):
        user_1 = unique_ids[p1_idx]
        for p2_idx in np.arange(p1_idx + 1, len(unique_ids)):
            user_2 = unique_ids[p2_idx]
            score = similarity_function(user_1, user_2)
            similarity_matrix[idx] = [user_1, user_2, score]
            idx += 1
            print("Completed:  ", idx, "/", int(val))
    return similarity_matrix


def recommend(person1, matches, similarity_function):
    """Computes a similarity score for two users given a specific similarity function."""
    similarity_matrix = np.zeros([int(len(matches)), 2])
    idx = 0
    for p2_idx in np.arange(len(matches)):
        user_2 = matches[p2_idx]
        if similarity_function == "spearman_correlation":
            score = spearman_similarity(person1, user_2)
        elif similarity_function == "pearson_correlation":
            score = pearson_correlation(person1, user_2)
        elif similarity_function == "euclidean_similarity":
            score = euclidean_similarity(person1, user_2)
        similarity_matrix[idx] = [user_2, score]
        idx += 1
        print("Completed:  ", idx, "/", int(len(matches)))
    return similarity_matrix
