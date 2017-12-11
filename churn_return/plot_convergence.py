import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
f = open("convergence_data.txt","r")
lines = []
for line in f:
    sp = line.split()
    sp[0] = int(sp[0])
    sp[1] = float(sp[1])
    sp = sp[:-1]
    lines += [sp]

data = np.array(lines)
sn.set(font_scale=2)
plt.plot(data[::,0],data[::,1])
plt.yscale("log")
plt.title("Log-Likelihood of Training Sequences Under Hidden Markov Model")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()
