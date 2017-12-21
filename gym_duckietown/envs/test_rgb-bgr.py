import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

# pyglet.gl.lib.GLException: b'invalid operation'
from scipy import misc

face = misc.imread('road.png')

print(face.shape)

fig = plt.figure(figsize=(1, 2))

fig.add_subplot(1, 2, 1)

plt.imshow(face)

fig.add_subplot(1, 2, 2)
plt.imshow(face[:, :, ::-1])

plt.show()
