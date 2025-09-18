import pickle
import cv2
import matplotlib.pyplot as plt

with open("CVPR.pkl", "rb") as f:
    saved_list = pickle.load(f)

# flip and convert to RGB
for i in range(3):
    saved_list[i] = cv2.flip(saved_list[i], -1)
    saved_list[i] = cv2.cvtColor(saved_list[i], cv2.COLOR_BGR2RGB)

# overlay the two images
plt.figure(figsize=(10, 5))
plt.imshow(saved_list[0])
plt.imshow(saved_list[2], alpha=0.5)
plt.show()