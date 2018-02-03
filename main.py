import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

import math
import time

oImg = None
count = 0
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

old_left_line = []
old_right_line = []

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global count
    global old_left_line, old_right_line
    left_lane = []
    right_lane = []
    print("========================"+str(count)+"=============")
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient = (y1-y2)/(x1-x2)
            print(str(gradient))
            if gradient < -0.5 and gradient > -0.9: # Note: (0, 0) is at top-left
                left_lane.append([x1,y1,x2,y2, gradient])
            elif gradient > 0.4 and gradient < 0.8:
                right_lane.append([x1,y1,x2,y2, gradient])
    left_gradient_avg = np.average([line[4] for line in left_lane])
    right_gradient_avg = np.average([line[4] for line in right_lane])
    print(str([left_gradient_avg, right_gradient_avg]))
    if math.isnan(left_gradient_avg):
       left_bottom_x, left_bottom_y, left_top_x, left_top_y = old_left_line
    else:
       left_x_avg = np.average([line[0] for line in left_lane] + [line[2] for line in left_lane])
       left_y_avg = np.average([line[1] for line in left_lane] + [line[3] for line in left_lane])
       left_bottom_x = int(left_x_avg - (left_y_avg - img.shape[0]) / left_gradient_avg)
       left_bottom_y = img.shape[0]
       left_top_y = 0
       left_top_x = int(left_x_avg - (left_y_avg - 0) / left_gradient_avg)
       
    if math.isnan(right_gradient_avg):
       right_bottom_x, right_bottom_y, right_top_x, right_top_y = old_right_line
    else:
       right_x_avg = np.average([line[0] for line in right_lane] + [line[2] for line in right_lane])
       right_y_avg = np.average([line[1] for line in right_lane] + [line[3] for line in right_lane])
       right_bottom_x = int(right_x_avg - (right_y_avg - img.shape[0]) / right_gradient_avg)
       right_bottom_y = img.shape[0]
       right_top_y = 0
       right_top_x = int(right_x_avg - (right_y_avg - 0) / right_gradient_avg)
       
    
    old_left_line = [left_bottom_x, left_bottom_y, left_top_x, left_top_y]
    old_right_line = [right_bottom_x, right_bottom_y, right_top_x, right_top_y]
    cv2.line(img, (left_bottom_x, left_bottom_y), (left_top_x, left_top_y), color, thickness)
    cv2.line(img, (right_bottom_x, right_bottom_y), (right_top_x, right_top_y), color, thickness)

            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def find_lane_pipeline(img):
    """
    `img` is the original image with lanes. It should be an RGB image with 3 channels
    The function returns a copy of the original image with red color annotation on the found lanes.
    """
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
    return final_img
    
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    global oImg
    global count
    oImg = np.copy(image)
    #mpimg.imsave('orig_frames/' + str(count) + '.jpg', oImg)
    result = find_lane_pipeline(image)
    #mpimg.imsave('issue_frames/' + str(count) + '.jpg', result)
    count+=1
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
yellow_output = 'test_videos_output/challenge.mp4'
output_dir = 'test_videos_output'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(yellow_output, audio=False)