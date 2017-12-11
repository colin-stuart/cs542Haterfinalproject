from hmmlearn import hmm
import numpy as np
import json
import joblib
import random
import warnings
warnings.filterwarnings("ignore")

f = open("hmm_data.txt","r")

print("Gathering Data:")
sequences = np.array([])
lengths = []
trained_on = []
withheld = []
num_withheld = 0
for line in f:
    data = json.loads(line)
    array = np.array(data)
    length = len(data)
    chances = random.random()
    if chances < 0.25:
        withheld += [data]
        num_withheld += 1
    else:
        trained_on += [data]
        sequences = np.concatenate([sequences,array])
        lengths += [length]
f.close()

print("Data gathered. " + str(num_withheld) + " samples withheld for testing.")
f = open("trained.txt", "w")
print(json.dumps(trained_on),file = f)
f.close()

f = open("test.txt","w")
print(json.dumps(withheld),file = f)
f.close()

print("Training Model:")
model = hmm.GaussianHMM(n_components=2,verbose=True,n_iter=1000).fit(sequences.reshape(-1,1),lengths)
print("Training Complete")
print("Saving model.")
joblib.dump(model, "hmm.pkl")
print("Program complete!")
