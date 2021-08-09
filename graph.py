import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style('whitegrid')
sns.set_palette('Paired')

x = np.array(['serial', 'serial_def', 'parallel'])
x_position = np.arange(len(x))

y_serial = np.array([2.53, 1.26, 1.24])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.bar(x_position, y_serial, width=0.8)

ax.legend(loc="lower left")
ax.set_xticks(x_position)
ax.set_xticklabels(x)
plt.show()

