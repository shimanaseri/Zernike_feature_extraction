import glob
import numpy as np
from jax import jit
import jax
import jax.numpy as jnp
import os
import skimage.io
import pandas as pd
from scipy.stats import skew
from skimage.color import rgb2gray
from mahotas.features import zernike_moments
import mahotas
import cv2  
import shutil
from skimage import img_as_ubyte

class ImagePreProcessor:
    def __init__(self,base_dir, verbose=False):
        self.verbose = verbose
        self.base_dir = base_dir
        self.processed_files = set()
        if not os.path.exists(self.base_dir):
          os.makedirs(self.base_dir) 

    def log(self, message):
        if self.verbose:
            print(message)
    
    def save_image(self, image, segment_name, file_name):
        # Create a directory for each segment if it does not exist
        segment_dir = os.path.join(self.base_dir, segment_name)
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)
        # Save the image
        file_path = os.path.join(segment_dir, file_name)
        skimage.io.imsave(file_path, img_as_ubyte(image))  # Save image as unsigned byte
        self.log(f"Saved {file_path}")

    def save_segment_images(self, image_segments, image_path):
        # Save each segment of the image in its corresponding folder
        file_name = os.path.basename(image_path)
        for segment_name, img in image_segments.items():
            self.save_image(img, segment_name, file_name)

    def numberRemoving(self, img):
        CONNECTIVITY = 4

        # HSV threshold for finding black pixels
        H_THRESHOLD = 179
        S_THRESHOLD = 255
        V_THRESHOLD = 150

        # read image
        img_height = img.shape[0]
        img_width = img.shape[1]

        # print("Image type:", type(img))
        # print("Image shape:", img.shape)

        # save a copy for creating resulting image
        result = img.copy()

        # convert image to grayscale
        gray = rgb2gray(np.array(img))
        gray = (gray * 255).astype(np.uint8)
        # print("Gray image shape:", gray.shape)
        # print("Gray image data type:", gray.dtype)

        # found the circle in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.7, minDist= 100, param1 = 48, param2 = 100, minRadius=70, maxRadius=100)
        circles = np.array([[[145, 145, 123]]])
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
            circles = np.round(circles[0, :]).astype("int")

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
        # self.log("Cutting circle in image...")
        img_shape = img.shape
        x_indices, y_indices = jnp.ogrid[:img_shape[0], :img_shape[1]]
        mask = (x_indices - 145) ** 2 + (y_indices - 145) ** 2 > 123**2
        img = jnp.array(img)  # Ensure img is a JAX array
        img = jnp.where(mask[:,:,None], 0, img)  # Apply mask, set to 0 where mask is True
        return self.numberRemoving(np.array(img))

    def process_image_segments(self, image_path):
        # self.log(f"Processing image segments for {image_path}...")
        img = skimage.io.imread(image_path)
        is_os = 'OS' in os.path.basename(image_path)

        img = jnp.array(img)  # Convert to jax.numpy array

        segments = {
            "AXIAL Curvature": img[10+120: 300+120, 15+434 : 302+434],
            "Corneal Thickness": img[690 - 330+27+120: 677+120,15+434 : 302+434],
            "Elevation (Front)": img[10+120: 690 - 330-60+120,690 - 330+29+434 : 690 - 330+302+14+434],
            "Elevation (Back)": img[690 - 330+27+120: 677+120,690 - 300+434: 434+677]
        }

        segment_features = {}
        for segment_name, segment_image in segments.items():
            self.log(f"Processing segment: {segment_name}")
            if is_os:
                segment_image = jnp.flip(segment_image, 1)
            preprocessed_image = self.cutCircle(segment_image)
            segment_features[segment_name] = preprocessed_image
            self.save_segment_images(segment_features, image_path)

        # self.log("Image segments processed.")
        return segment_features

    def process_dataset(self, dataset_dir):
        self.log(f"Processing dataset in {dataset_dir}...")
        data_records = []

        for subdir, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    file_path = os.path.join(subdir, file)
                    if file_path in self.processed_files:
                        self.log(f"Hello There")
                        continue
                    self.log(f"Processing file: {file_path}")
                    features = self.process_image_segments(file_path)

                    for segment_name, segment_image in features.items():
                        record = {
                            'File Path': file_path,
                            'Segment': segment_name,
                            'Image': segment_image  # Directly use the image associated with the segment
                        }
                        data_records.append(record)

        # self.log("Dataset processing complete.")
        print('Bye!')
        return data_records
    def zip_segments(self):
        # Create a zip file containing all segment directories
        output_zip = os.path.join(self.base_dir, 'segmented_images')
        shutil.make_archive(base_name=output_zip, format='zip', root_dir=self.base_dir)
        self.log(f"Created zip file {output_zip}.zip")
