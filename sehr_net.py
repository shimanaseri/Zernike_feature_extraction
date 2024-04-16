
!pip install scipy scikit-image cupy-cuda100
!pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install jax jaxlib
!pip install mahotas


import glob
import numpy as np
from jax import jit
import jax
import jax.numpy as jnp
import cv2  # Note: OpenCV still uses numpy arrays
import mahotas
import os
import shutil
from mahotas.features import zernike_moments
from scipy.stats import skew
from skimage.color import rgb2gray
import skimage.io
import pandas as pd
from jax import ops

class Zernike_ImageProcessor:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)
            
    
    def numberRemoving(self, img):
        CONNECTIVITY = 4

        # HSV threshold for finding black pixels
        H_THRESHOLD = 179
        S_THRESHOLD = 255
        V_THRESHOLD = 150

        # read image
        img_height = img.shape[0]
        img_width = img.shape[1]

        # save a copy for creating resulting image
        result = img.copy()

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # found the circle in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.7, minDist= 100, param1 = 48, param2 = 100, minRadius=70, maxRadius=100)
        circles = jnp.array([[[145, 145, 123]]])
        # draw found circle, for visual only
        circle_output = img.copy()

        # check if we found exactly 1 circle
        num_circles = len(circles)
        # print("Number of found circles:{}".format(num_circles))
        if (num_circles != 1):
            print("invalid number of circles found ({}), should be 1".format(num_circles))
            exit(0)

        # save center position and radius of found circle
        circle_x = 0
        circle_y = 0
        circle_radius = 0
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = jnp.round(circles[0, :]).astype("int")

            for (x, y, radius) in circles:
              circle_x, circle_y, circle_radius = (int(x), int(y), int(radius))  # Convert to integers
              cv2.circle(circle_output, (circle_x, circle_y), circle_radius, (255, 0, 0), 4)

                # print("circle center:({},{}), radius:{}".format(x,y,radius))

        # keep a median filtered version of image, will be used later
        median_filtered = cv2.medianBlur(img, 19)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



        # Convert the threshold values to NumPy arrays using 'onp.array'
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([H_THRESHOLD, S_THRESHOLD, V_THRESHOLD])

        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_val, upper_val)

        # find connected components
        components = cv2.connectedComponentsWithStats(mask, CONNECTIVITY, cv2.CV_32S)

        # apply median filtering to found components
        num_components = components[0]
        # print("Number of found connected components: " +str components)

        labels = components[1]
        stats = components[2]
        for i in range(1, num_components):
            left = stats[i, cv2.CC_STAT_LEFT] - 10
            top = stats[i, cv2.CC_STAT_TOP] - 10
            width = stats[i, cv2.CC_STAT_WIDTH] + 10
            height = stats[i, cv2.CC_STAT_HEIGHT] + 10

            # iterate each pixel and replace them if
            # they are inside circle
            for row in range(top, top + height + 1):
                for col in range(left, left + width + 1):
                    dx = col - circle_x
                    dy = row - circle_y
                    if (dx * dx + dy * dy <= circle_radius * circle_radius):
                        result[row, col] = median_filtered[row, col]
        return result
    def cutCircle(self, img):
        self.log("Cutting circle in image...")
        img_shape = img.shape
        x_indices, y_indices = jnp.ogrid[:img_shape[0], :img_shape[1]]
        mask = (x_indices - 145) ** 2 + (y_indices - 145) ** 2 > 123**2
        img = jnp.array(img)  # Ensure img is a JAX array
        img = jnp.where(mask[:,:,None], 0, img)  # Apply mask, set to 0 where mask is True
        return self.numberRemoving(img)

    def compute_features(self, preprocessed_image, radius, degree, iscolor=False):
        self.log("Computing features...")
        gray_image = rgb2gray(jnp.array(preprocessed_image))  # Ensure compatibility with skimage
        if gray_image.dtype != bool:
            gray_image = gray_image > gray_image.mean()
        zernike_features = zernike_moments(gray_image, radius, degree=degree)
        self.log("Zernike moments computed.")

        if iscolor:
            self.log("Computing color features...")
            color_features = {}
            # Conversion to numpy array may be necessary for compatibility with non-JAX operations
            preprocessed_image_np = np.array(preprocessed_image)
            for color, channel in zip(['red', 'green', 'blue'], np.rollaxis(preprocessed_image_np, axis=-1)):
                mean = jnp.mean(channel)
                std = jnp.std(channel)
                skewness = skew(channel.flatten())  # This may require a JAX-compatible skew function
                color_features[color] = {'mean': mean, 'std': std, 'skewness': skewness}
            self.log("Color features computed.")
            return {
                'zernike_moments': zernike_features,
                'color_features': color_features
            }
        else:
            return {
                'zernike_moments': zernike_features
            }

    def process_image_segments(self, image_path, iscolor=False):
        self.log(f"Processing image segments for {image_path}...")
        img = skimage.io.imread(image_path)
        img = jnp.array(img)  # Convert to jax.numpy array

        segments = {
            "AXIAL Curvature": img[10: 300, 15 : 302],
            "Corneal Thickness": img[10 : 300,690 - 300: 677],
            "Elevation (Front)": img[690 - 330+27: 677,15 : 302],
            "Elevation (Back)": img[690 - 330+27: 677,690 - 300: 677] 
        }

        segment_features = {}
        for segment_name, segment_image in segments.items():
            self.log(f"Processing segment: {segment_name}")
            preprocessed_image = self.cutCircle(segment_image)
            features = self.compute_features(preprocessed_image, radius=123, degree=4, iscolor=iscolor)
            segment_features[segment_name] = features

        self.log("Image segments processed.")
        return segment_features

    def process_dataset(self, dataset_dir, iscolor=False):
        self.log(f"Processing dataset in {dataset_dir}...")
        data_records = []

        for subdir, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    file_path = os.path.join(subdir, file)
                    self.log(f"Processing file: {file_path}")
                    features = self.process_image_segments(file_path, iscolor=iscolor)

                    for segment_name, segment_features in features.items():
                        record = {
                            'File Path': file_path,
                            'Segment': segment_name,
                            'Zernike Moments': segment_features['zernike_moments'],
                        }
                        if iscolor:
                            for color, color_features in segment_features['color_features'].items():
                                for feature_name, feature_value in color_features.items():
                                    record[f'{color}_{feature_name}'] = feature_value

                        data_records.append(record)

        df = pd.DataFrame(data_records)
        self.log("Dataset processing complete.")
        return df
