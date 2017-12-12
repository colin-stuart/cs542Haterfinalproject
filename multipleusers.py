import numpy as np
from Project import MatchAlgorithmFunctions as maf
from Project import LogisticRegression as log_reg
import json
from sklearn.model_selection import train_test_split
#from Project import AnalyzeFeatures as af


# Pull users information from user_data.txt

f = open("user_data.txt","r")
user_data = json.loads(f.read())
user_data = np.array(user_data)  # converting to array

# Pull match information from user_matches.txt

f = open("user_matches.txt","r")
true_matches = json.loads(f.read())
true_matches = np.array(true_matches)   # converting to array
true_matches = true_matches[:40000]   # to have only 40,000 users to avoid a MemoryError

print("Printing true matches..")   # debugging purposes
print(true_matches)   # debugging purposes

# creates the array of 0 or 1 depending on if the person liked or disliked the other user
swiped = np.zeros((true_matches.shape[0], 1))
swiped[np.where(true_matches == 'Like'), 0] = 1

# Pulls the swiped peoples' data
true_match_ids = true_matches[:, 0].astype(int)

print("Printing true match IDs...")  # debugging purposes
print(true_match_ids)  # debugging purposes

# The orders of the IDs are matching the IDs obtained from true_matches.txt

f = open("user_true_match_data.txt","r")
true_match_data = json.loads(f.read())
true_match_data = np.array(true_match_data)
true_match_data = true_match_data[:40000]    # use only 40,000 users to avoid a MemoryError

print("Printing true match data..")    # debugging purposes
print(true_match_data)    # debugging purposes

# User pictures for extroversion score

f = open("user_pictures.txt", "r")
num_pic = 0   # keeps a count of the number of pictures per user. Max = 5 pictures, minimum = 0.
num_pic_mat = []   # an array to keep track of the number of pictures for each user
for line in f:
    num_pic = line.count("Picture")   # counts number of pictures
    num_pic_mat += [num_pic]
f.close()

# For similarity score
# similarity_score = []
# for i in range(len(true_matches)):
#     similarity_score += [af.cosine_similarity(int(true_matches[i][0]), int(true_matches[i][1]))]
# print("Similarity score:", similarity_score)

##################################################################################
################ LOGISTIC REGRESSION DATA AND FUNCTION CALL BEGINS ###############

# params: N x M matrix with N being the number of people this user previously matched with and M being the features
# Features: age, gender_pred, extroversion_score, attractiveness

pref_female, pref_male = maf.gender_pref(true_match_data[:, 2].astype(int))
gender_pref_array = np.zeros(true_match_data.shape[0])
gender_pref_array[np.where(true_match_data[:,2].astype(int) == 0)] = pref_male
gender_pref_array[np.where(true_match_data[:,2].astype(int) == 1)] = pref_female
params = np.zeros((true_match_data.shape[0], 4))
params[:, 0] = true_match_data[:, 1].astype(int)
params[:, 1] = gender_pref_array
params[:, 2] = true_match_data[:, 2].astype(float)
params[:, 3] = maf.extroversion_score(num_pic_mat[:40000])    # 40,000 users
#params[:, 4] = similarity_score


# Check the X (params) and Y (swiped)
print("Printing params")
params = params[:40000]
print(params)

print("printing swiped..")
print(swiped)

# LOGISTIC REGRESSION CALL

# Splitting into test and train (20% and 80%) - 8000 test and 32000 train

X_train, X_test, y_train, y_test = train_test_split(params, swiped, test_size=0.20)

# For debugging

# print("X_train.shape:", X_train.shape)
# print("Y_train.shape:", y_train.shape)
# print("X_test.shape:", X_test.shape)
# print("y_test.shape:", y_test.shape)

# Logistic Regression - calling for X_train and y_train first
accuracy = log_reg.modified_model(X_train, y_train)
print(accuracy)

# Logistic Regression - calling for X_train and y_train first
accuracy = log_reg.modified_model(X_test, y_test)
print(accuracy)







