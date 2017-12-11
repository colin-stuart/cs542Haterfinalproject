import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
array = np.array([[16701,662],
    [2062,6410]])



sn.set(font_scale=2.5)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(array,cmap="GnBu")
plt.title('Hidden Markov Model Confusion Matrix',y=1.08)
fig.colorbar(cax)
ax.set_xticklabels([''] + ["0","1"])
ax.set_yticklabels([''] + ["0","1"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
