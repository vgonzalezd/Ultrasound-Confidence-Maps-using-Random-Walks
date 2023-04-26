import matplotlib.pyplot as plt
import scipy.io

from confidence_map_numpy.confidence_map import confidence_map

# Load neck data and call confidence estimation for B-mode with default parameters
img = scipy.io.loadmat('data/neck.mat')['img']
alpha, beta, gamma = 2.0, 90, 0.03
map_ = confidence_map(img, alpha, beta, gamma)

# Display neck images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(map_, cmap='gray')
plt.axis('off')

# Load femur data and call confidence estimation for B-mode with default parameters
img = scipy.io.loadmat('data/femur.mat')['img']
alpha, beta, gamma = 2.0, 90, 0.06
map_ = confidence_map(img, alpha, beta, gamma)

# Display femur images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(map_, cmap='gray')
plt.axis('off')

plt.show()
