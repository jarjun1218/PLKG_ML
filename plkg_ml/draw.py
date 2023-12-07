import matplotlib.pyplot as plt
import numpy as np

losses = np.load('plkg_ml/encoder/losses.npy')
plt.plot(losses[0],losses[1])
plt.show()