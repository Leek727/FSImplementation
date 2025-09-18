import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

with open("CVPR.pkl", "rb") as f:
    saved_list = pickle.load(f)

for i in range(len(saved_list)):
    # flip vertically and horizontally
    saved_list[i] = cv2.flip(saved_list[i], -1)

    # change color to RGB
    if saved_list[i].shape == (480, 640, 3):
        saved_list[i] = cv2.cvtColor(saved_list[i], cv2.COLOR_BGR2RGB)

# crop to cut out homography artifact
saved_list[2] = saved_list[2][0:400, 0:550]
saved_list[3] = saved_list[3][0:400, 0:550]
saved_list[4] = saved_list[4][0:400, 0:550]

# threshold with confidence map
saved_list[3] = saved_list[3] * (saved_list[4]**2 > 0.00003)
#* (abs(saved_list[4]) > 0.004)

# custom colormap to match the color palette of the paper
colors = ['purple', 'blue', 'teal', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

# show side by side, remove axis
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(saved_list[2])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(saved_list[3], cmap=custom_cmap)
plt.axis('off')
plt.show()