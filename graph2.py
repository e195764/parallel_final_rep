import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#graph2
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Paired')

x = np.array(['serial', 'serial_def', 'parallel'])
x_position = np.arange(len(x))

y_ac = np.array([0.80, 0.85, 0.79])
y_f = np.array([0.67, 0.73, 0.79])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.bar(x_position, y_ac, width=0.4, label='accuracy')
ax.bar(x_position + 0.4, y_f, width=0.4, label='f-value')

ax.legend(loc="lower left")
ax.set_xticks(x_position + 0.2)
ax.set_xticklabels(x)
ax.set_xlabel('index')
plt.show()
