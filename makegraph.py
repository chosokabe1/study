import numpy as np 
import matplotlib.pyplot as plt

x  = np.arange(-5, 5, 0.1)
y = np.sin(x)

plt.plot(x,y)

data = [2, 4, 6, 3, 5, 8, 4, 5]
plt.plot(data)

plt.show()