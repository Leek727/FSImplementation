import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import LinearSegmentedColormap

calibration_pkl_path = "calibration_files.pkl"
inference_pkl_path = "pokemon.pkl"
parameters = [11.87780144, 13.68435359] 
crop_size = (440, 565)
training_flag = int(input("Calibration (1) or Inference (0)?: "))
percentile_threshold = 60

# custom colormap to match the color palette of the paper
colors = ['purple', 'blue', 'teal', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
custom_cmap.set_bad(color='white')

def get_test_img():
    """Returns I1 and aligned I2 both flipped and converted to RGB"""
    with open(inference_pkl_path, "rb") as f:
        saved_list = pickle.load(f)
    img1 = saved_list[0]
    img2 = saved_list[2]
    img1 = flip_and_convert(img1)
    img2 = flip_and_convert(img2)
    return img1, img2

# --------------------- preprocessing ---------------------
def preprocess(img):
    """Returns KxK box filter + gaussian blurred image. Takes in RGB image."""
    K = 21 # box filter size
    G = 11 # gaussian blur 11 pixels standard deviation

    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # subtract box filter to remove non-uniform background lighting
    img = img - cv2.boxFilter(img, -1, (K, K))

    # noise attenuation - arbitrary kernel size
    img = cv2.GaussianBlur(img, (3, 3), G)
    return img

def flip_and_convert(img):
    """Flips and converts image to RGB"""
    img = cv2.flip(img, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def test_preprocess():
    img, _ = get_test_img()
    
    img_processed = preprocess(img)

    # show side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_processed)
    plt.axis('off')
    plt.show()


# --------------------- Approximate Image Derivatives ---------------------
def approximate_image_derivatives(I1, I2):
    """I1 and I2 are aligned and preprocessed images. Returns ~Is and Laplacian of ~I per pixel."""

    Is = I2 - I1 # image 2 is the one getting transformed so subtract image 1 from image 2
    combined_image = (I1 + I2) / 2
    Laplacian = cv2.Laplacian(combined_image, -1)
    return Is, Laplacian

def test_approximate_image_derivatives():
    """Function to visualize the approximate image derivatives"""
    img1, img2 = get_test_img()
    img1 = flip_and_convert(img1)
    img2 = flip_and_convert(img2)
    img1 = preprocess(img1)
    img2 = preprocess(img2)

    # show 4x4, img 1, img2, Is, Laplacian
    Is, Laplacian = approximate_image_derivatives(img1, img2)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1 (preprocessed)')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Image 2 (preprocessed)')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(Is)
    plt.title('Is = I1 - I2')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(Laplacian)
    plt.title('Laplacian((I1+I2)/2)')
    plt.axis('off')
    plt.show()



# --------------------- Depth Estimation ---------------------
def estimate_depth(Is, LapI, a, b):
    """Returns the computed depth map given Is, Laplacian, and calibrated parameters a, b"""
    L = 21 # box filter size
    
    # calculate patchwise soln in two steps
    Z_numerator = (a * LapI * (b * LapI + Is))
    Z_numerator = cv2.boxFilter(Z_numerator, -1, (L, L))

    Z_denominator = (b * LapI + Is)**2
    Z_denominator = cv2.boxFilter(Z_denominator, -1, (L, L))

    Z = Z_numerator / Z_denominator
    return Z


# --------------------- Parameter Calibration ---------------------
if training_flag:
    with open(calibration_pkl_path, "rb") as f:
            calibration_data = pickle.load(f)
else:
    calibration_data = []

def conf_from_img(img1, img2):
    """Takes in img1 and aligned img2. Assumes images are preprocessed."""

    img1 = cv2.GaussianBlur(img1, (5,5), 11)
    img2 = cv2.GaussianBlur(img2, (5,5), 11)
    conf_map = (img1 - img2)**2
    threshold = np.percentile(conf_map, percentile_threshold)
    conf_map = conf_map > threshold

    return conf_map

def loss_fn(params):
    """Returns squared norm error between Z_true and Z_pred given [a,b]"""
    a, b = params
    print(params)

    loss = 0
    for pair in calibration_data:
        img1 = pair[0]["Img"]
        img2 = pair[1]["Img"]
        Z_true = pair[0]["Loc"] / 2500000 + 0.4 # convert to m where 25000 steps is 1 cm, then add offset of 0.4 m

        img1 = flip_and_convert(img1)
        img2 = flip_and_convert(img2)
        img1 = preprocess(img1)
        img2 = preprocess(img2)
      
        Is, Laplacian = approximate_image_derivatives(img1, img2)

        Z_pred = estimate_depth(Is, Laplacian, a, b)

        """
        # they do not use confidence threshold during calibration
        if threshold_flag:
            print(f"Note: Confidence threshold is on in loss_fn")
            confidence_map = conf_from_img(img1, img2)
            confidence_map = (confidence_map > 0.004)
            Z_pred = confidence_map * Z_pred

            # crop to remove homography artifact 
            Z_pred = Z_pred[0:crop_size[0], 0:crop_size[1]]
            
            # squared norm
            # threshold Z_true with confidence map so that its zero if confidence is low and true depth if confidence is high
            Z_true = Z_true * confidence_map
            Z_true = Z_true[0:crop_size[0], 0:crop_size[1]]
            loss += np.sum((Z_true - Z_pred)**2)
        """

        # crop to remove homography artifact 
        Z_pred = Z_pred[0:crop_size[0], 0:crop_size[1]]
            
        # squared norm
        loss += np.sum((Z_true - Z_pred)**2)
        
    print(loss)

    return loss

def calibrate_parameters():
    """Returns calibrated parameters [a,b]. Reads dataset from calibration_pkl_path var."""
    
    params = parameters
    mymin = minimize(loss_fn, params, method='BFGS')
    print(mymin)
    return mymin.x


def test_calibrate_parameters():
    """Test the calibrate_parameters function"""
    params = calibrate_parameters()
    print(params)


# show side by side view
def validate_depth():
    img1, img2 = get_test_img()
    img1 = preprocess(img1)
    img2 = preprocess(img2)
    Is, Laplacian = approximate_image_derivatives(img1, img2)

    a,b = parameters

    # Call the debug function to inspect values
    Z = estimate_depth(Is, Laplacian, a, b)

    # threshold with estimated confidence map
    Z *= conf_from_img(img1, img2)

    Z = Z[0:crop_size[0], 0:crop_size[1]]
    # for white background
    Z_masked = np.ma.masked_where(Z == 0, Z)


    # --------------------- load ground truth ---------------------
    # load ground truth confidence map
    with open(inference_pkl_path, "rb") as f:
        saved_list = pickle.load(f)
    true_confidence_map = (cv2.flip(saved_list[4], -1))**2
    threshold = np.percentile(true_confidence_map, percentile_threshold)
    true_confidence_map = true_confidence_map > threshold
    

    ground_truth = saved_list[3]
    # flip and threshold
    ground_truth = cv2.flip(ground_truth, -1) * true_confidence_map
    # for white background
    ground_truth = np.ma.masked_where(ground_truth == 0, ground_truth)
    ground_truth = ground_truth[0:crop_size[0], 0:crop_size[1]]


    # show side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, cmap=custom_cmap)
    plt.subplot(1, 2, 2)
    plt.imshow(Z_masked, cmap=custom_cmap)
    plt.show()


if training_flag:
    test_calibrate_parameters()
else:
    validate_depth()