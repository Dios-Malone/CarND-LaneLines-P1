import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from main import *

img = mpimg.imread('issue_images/issueImg.jpg')
img = img[:,:,:3]
print(img.shape)
#Parameters for canny process:
canny_low_threshold = 50
canny_high_threshold = 150

#Parameters for Hough transform
hough_rho = 1
hough_theta = (np.pi/180)
hough_threshold = 50
hough_min_line_len = 30
hough_max_line_gap = 30

#Parameters for interested area selection
img_width = img.shape[1]
img_height = img.shape[0]
p_bottom_left = (120, img_height)
p_top_left = (int(img_width * 0.45), int(img_height * 0.59))
p_top_right = (int(img_width * 0.55), int(img_height * 0.59))
p_bottom_right = (img_width, img_height)
vertices = np.array([[p_bottom_left,p_top_left, p_top_right, p_bottom_right]], dtype=np.int32)

#pipeline start:
# 1. create a copy of grayscaled image of original image
gray_img = grayscale(np.copy(img))

# 2. apply Gaussian blur
blur_img = gaussian_blur(gray_img, 5)

# 3. Detect edges by Canny function
canny_img = canny(blur_img, canny_low_threshold, canny_high_threshold)

# 4. Mask edge image by selecting interested area
interested_img = region_of_interest(canny_img, vertices)

# 5. Detect lines by using Hough transform and create lines image
line_img = hough_lines(interested_img, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)

interested_line_img = region_of_interest(line_img, vertices)

# 6. Draw the detected lines onto the origianl image
final_img = weighted_img(interested_line_img, img)

plt.imshow(final_img)
plt.show()