import src.strauss as strauss
import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(1, 90, 400)
y = x
# y = np.negative(y)

plt.xlabel('Parameter #1 (x)')
plt.ylabel('Parameter #2 (y)')
plt.plot(x,y)
plt.show()

strauss.sonify(y, x, style='my_style.yml')
strauss.save('soni.wav')
strauss.close()