import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

with open("CVPR.pkl", "rb") as f:
    saved_list = pickle.load(f)

# flip all imgs
for i in range(5):
    saved_list[i] = cv2.flip(saved_list[i], -1)
    
def preprocess(img):
    """Returns KxK box filter + gaussian blurred image. Takes in RGB image."""
    K = 21 # box filter size
    G = 11 # gaussian blur 11 pixels standard deviation


    # subtract box filter to remove non-uniform background lighting
    scalar = 1 / K**2
    img = img - cv2.boxFilter(img, -1, (K, K))

    # noise attenuation - kernel size not specd in paper so change later
    img = cv2.GaussianBlur(img, (3, 3), G)
    return img


def conf_from_img(img1, img2):
    """Takes in img1 and aligned img2"""
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    img1 = cv2.GaussianBlur(img1, (5,5), 11)
    img2 = cv2.GaussianBlur(img2, (5,5), 11)

    #C = (img2 - img1) ** 2
    C = img1 - img2


    #img = (img1 + img2) * 2
    #img = cv2.GaussianBlur(img, (3,3), 11)
    #Laplacian = cv2.Laplacian(img, -1)
    # box filter
    #Laplacian = cv2.boxFilter(Laplacian, -1, (7,7))

    return C

ground_truth = saved_list[4]
Is = conf_from_img(saved_list[0], saved_list[2])

ground_truth = ground_truth[0:400, 0:550]
Is = Is[0:400, 0:550]

# show side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

threshold = np.percentile(ground_truth**2, 40)
ground_truth = ground_truth**2 > threshold
plt.imshow(ground_truth)
plt.subplot(1, 2, 2)

# discard 40% least confident pixels
Is_squared = Is**2
# get 40th percentile value for thresholding
threshold = np.percentile(Is_squared, 40)
Is_squared = Is_squared > threshold

plt.imshow(Is_squared)
plt.show()