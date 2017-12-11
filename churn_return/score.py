import json
from hmmlearn import hmm
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

trained_f = open("trained.txt","r")
test_f = open("test.txt","r")

trained = json.loads(trained_f.read())
test = json.loads(test_f.read())
trained_f.close()
test_f.close()

trained_lengths = []
test_lengths = []

trained_sequence = np.array([])
test_sequence = np.array([])

print("Gathering trained sequences")
trained_map = {}
emm = 0
for i in range(len(trained)):
    if len(trained[i]) < 3:
         continue
    trained_sequence = np.concatenate([trained_sequence, np.array(trained[i][:-2])])
    trained_lengths += [len(trained[i][:-2])]
    trained_map[emm] = i
    emm += 1
print("training emm:" + str(emm))

test_map = {}
emm = 0
print("Gathering test sequences")
for i in range(len(test)):
    if len(test[i]) < 3:
        continue
    test_sequence = np.concatenate([test_sequence,np.array(test[i][:-2])])
    test_lengths += [len(test[i][:-2])]
    test_map[emm] = i
    emm += 1
print("test emm:" + str(emm))

trained_sequence = trained_sequence.reshape(-1,1)
test_sequence = test_sequence.reshape(-1,1)
print("loading model")
model = joblib.load("hmm.pkl")
print(len(trained_lengths))
trained_score = model.score_samples(trained_sequence, trained_lengths)
test_score = model.score_samples(test_sequence,test_lengths)
print(trained_score)
print(len(trained_score[1]))

def baseline_accuracy(model,sequences,emissions,map_):
    correct = 0
    total = 0
    incorrect_1 = 0
    incorrect_0 = 0
    correct_1 = 0
    correct_0 = 0
    print("beginning scoring")
    for i in range(len(emissions)):
        full_prediction = model.predict(np.array(sequences[map_[i]]).reshape(-1,1))
        next_predicted = 0
        if next_predicted == full_prediction[-2]:
            if next_predicted == 1:
                correct_1 += 1
            else:
                correct_0 += 1
            correct += 1
        else:
            if next_predicted == 1:
                incorrect_1 += 1
            else:
                incorrect_0 += 1
        total += 1

    print("incorrect 1 predictions:" + str(incorrect_1))
    print("incorrect 0 predictions:" + str(incorrect_0))
    print("correct 1 predictions:" + str(correct_1))
    print("correct 0 predictions:" + str(correct_0))
    return correct/float(total)

def predict_next(model, sequences,emissions,map_):
    f = open("correct.txt","w")
    f2 = open("incorrect.txt","w")
    correct = 0
    total = 0
    incorrect_1 = 0
    incorrect_0 = 0
    correct_1 = 0
    correct_0 = 0
    print("beginning scoring")
    for i in range(len(emissions)):
        full_prediction = model.predict(np.array(sequences[map_[i]]).reshape(-1,1))
        prediction = full_prediction[:-2]
        transition = model.transmat_
        chance_of_zero = emissions[i][-1]
        chance_of_one = 1-chance_of_zero
        chance_of_zero_next = chance_of_zero*transition[0][0] + chance_of_one*transition[1][0]
        if chance_of_zero_next > 0.5:
            next_predicted = 0
        else:
            next_predicted = 1
        if next_predicted == full_prediction[-2]:
            if next_predicted == 1:
                correct_1 += 1
            else:
                correct_0 += 1
            correct += 1
            print(json.dumps(sequences[i][:-1]) + "prediction: " + str(next_predicted),file = f)
        else:
            print(json.dumps(sequences[i][:-1]) + "prediction: " + str(next_predicted),file = f2)
            if next_predicted == 1:
                incorrect_1 += 1
            else:
                incorrect_0 += 1
        total += 1

    print("incorrect 1 predictions:" + str(incorrect_1))
    print("incorrect 0 predictions:" + str(incorrect_0))
    print("correct 1 predictions:" + str(correct_1))
    print("correct 0 predictions:" + str(correct_0))
    f.close()
    f2.close()
    return correct/float(total)

def assemble_emmissions(emmissions,lengths):
    ret = []
    emmissions = emmissions[::,:1]
    emmissions = list(emmissions.reshape(1,-1))
    emmissions = emmissions[0]
    for i in lengths:
        ret += [emmissions[:i]]
        emmissions = emmissions[i:]
    return ret

print("scoring")
emissions = assemble_emmissions(trained_score[1],trained_lengths)
print("Training Score:" + str(predict_next(model,trained,emissions,trained_map)))
emissions = assemble_emmissions(test_score[1],test_lengths)
print("Test Score:" + str(predict_next(model,test,emissions,test_map)))

emissions = assemble_emmissions(trained_score[1],trained_lengths)
print("Training Baseline:" + str(baseline_accuracy(model,trained,emissions,trained_map)))
emissions = assemble_emmissions(test_score[1],test_lengths)
print("Test Baseline:" + str(baseline_accuracy(model,test,emissions,test_map)))
