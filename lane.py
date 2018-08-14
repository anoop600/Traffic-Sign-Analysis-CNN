
# coding: utf-8

# # Self-Driving Car
# 
# ## Finding Lane Lines
# ***
# The goal of this project is to highligh lane lines in a number of still images and short videos. This is achived with an image processing pipline.

# In[33]:


#########################DATABASE################################
import MySQLdb as my
db = my.connect("127.0.0.1","root","","pythondb")
cursor = db.cursor()
sql = "UPDATE `task` SET value = 0 WHERE slno=1;"
cursor.execute(sql)
db.commit()

#########################END-DATABASE#######################################


# ## Import Packages

# In[34]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, glob
import math
get_ipython().run_line_magic('matplotlib', 'inline')

# for the movies
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# ## Reading in Images
# ***
# To get started, I read in the test images and plotted them as is with code from the lesson. There are a few things that may cause difficulties such as some some small horizontal lines that might make it through the processing steps, though the images are mostly open, straight road under reasonable light. 
# ***

# In[35]:


#reading in and plotting raw images
images = os.listdir("test_images/")

test_images = []
for img in images:
    test_images.append(mpimg.imread("test_images/" + img))
    plt.figure()
    plt.imshow(mpimg.imread("test_images/" + img))


# ## Processing functions
# 
# ***
# Here are the processing functions used to build the pipline to identify the lanes.
# ***

# In[36]:


#=============================================================
# functions required for pipline

# convert to grayscale, call plt.imshow(gray, cmap='gray') to plot.
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# canny edge detection low high thresholds likely between 0 and 200, roughly 100 or so apart
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# blur to highlight straight lines, kernel size is positie and odd number. try 5 - 25
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# ROI mask veritices is a np.array, using dimentions from lesson  
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# added defaults 
def weighted_img(img, initial_img, α=0.7, β=0.3, λ=0):
    """
    `img` is the output of the hough_lines(),blacked out. 
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Used for comparison
def hough_lines(img, rho = 1, theta = 0.0017, threshold = 20, min_line_len = 20, max_line_gap = 280):
    """
    'img' should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # from draw lines function
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 15)
    return line_img


# ***
# Start of processing image
# ***

# In[37]:


# test pipline 
def test_lines(img):
    #img1 = select_rgb_white_yellow(img) # source [1]
    img2 = grayscale(img)
    img3 = gaussian_blur(img2, kernel_size = 5)
    img4 = canny(img3, 50, 150)
    
    # define the roi extents
    left_bottom = [0, 539]
    right_bottom = [900, 539]
    apex = [475, 320]
    vertices = np.array([[left_bottom, right_bottom, apex]], dtype=np.int32)
    
    # cut image to roi
    img_roi = region_of_interest(img4, vertices)
    
    # hough transform from helper functions
    img6 = hough_lines(img_roi)
    img7 = weighted_img(img6, img)
    return img7

# loop through the test images and plot result
for img in test_images:
    plt.figure()
    plt.imshow(test_lines(img))


# ## Finishing the lines
# ***
# The second function plots the lines on an imput image, or, if the line is equal to 
# the number 9999, the result I have if there are NaN objects in the first function, 
# it just returns the original. This is not ideal, but was nessesary to get through
# processing the whole second video.
# ***

# In[38]:


# makes 1 average-ish line for each side by averaging the points and 
def average_lines(line_img):
    # hough lines function from lesson 
    lines = cv2.HoughLinesP(line_img, rho = 1, theta = 0.0017, threshold = 20, minLineLength = 20, maxLineGap=280)
    # initilize emtpy objects

    # initilize all objects for two lines defined by two points each
    # consant values though, for y1 and y2 as they will be middle and 
    #  bottom of image 
    r_x1_points = []
    r_x2_points = []
    r_y1_points = []
    r_y2_points = []
    l_x1_points = []
    l_x2_points = []
    l_y1_points = []
    l_y2_points = []

    for line in lines:
        # calculate individual line slope
        slope = ((line[0, 3] - line[0, 1]) / ( line[0, 2] - line[0, 0]))
        if slope > 0:
            r_x1_points.append(line[0, 0])
            r_x2_points.append(line[0, 2])
            r_y2_points.append(line[0, 3])
            r_y1_points.append(line[0, 1])
        if slope < 0: 
            l_x1_points.append(line[0, 0])
            l_x2_points.append(line[0, 2])
            l_y2_points.append(line[0, 3])
            l_y1_points.append(line[0, 1])
        
    # average the points before re-calculating one slope for R and L line
    r_x1_ave = np.mean(r_x1_points)
    r_x2_ave = np.mean(r_x2_points)
    r_y1_ave = np.mean(r_y1_points)
    r_y2_ave = np.mean(r_y2_points)
    l_x1_ave = np.mean(l_x1_points)
    l_x2_ave = np.mean(l_x2_points)
    l_y1_ave = np.mean(l_y1_points)
    l_y2_ave = np.mean(l_y2_points)
    
    ave_r_slope = (r_y2_ave - r_y1_ave) / (r_x2_ave - r_x1_ave)
    ave_l_slope = (l_y2_ave - l_y1_ave) / (l_x2_ave - l_x1_ave)
    
    l_yint = l_y2_ave - ave_l_slope * l_x2_ave
    r_yint = r_y2_ave - ave_r_slope * r_x2_ave
    
    # geometry 
    # y values for both are assumed to be 320 and 539
    #  this will extend the lines 
    l_y1_final = 320
    r_y1_final = 320
    l_y2_final = 539
    r_y2_final = 539
    
    # left side!!!
    l_x1_final = (l_y1_final - l_yint) / ave_l_slope
    l_x2_final = (l_y2_final - l_yint) / ave_l_slope
    
    # strong side!!!
    r_x1_final = (r_y1_final - r_yint) / ave_r_slope
    r_x2_final = (r_y2_final - r_yint) / ave_r_slope
    
    if math.isnan(r_x1_ave) or math.isnan(r_x2_ave) or math.isnan(r_y1_ave) or math.isnan(r_y2_ave) or math.isnan(l_x1_ave) or math.isnan(l_x2_ave) or math.isnan(l_y1_ave) or math.isnan(l_y2_ave):
        return [9999]
    else:
        #organize into line object for input_lines to draw lines
        r_p1 = (int(r_x1_final), int(r_y1_final))
        r_p2 =  (int(r_x2_final), int(r_y2_final))
        l_p1 = (int(l_x1_final), int(l_y1_final))
        l_p2 = (int(l_x2_final), int(l_y2_final))
        
        input_lines = [r_p1, r_p2, l_p1, l_p2]

        return input_lines

def final_lane_lines(img, lines):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines != [9999]:
        # write first line
        cv2.line(line_img, lines[0], lines[1], [255, 0, 0], 15)
        cv2.line(line_img, lines[2], lines[3], [255, 0, 0], 15)   
        
        return line_img
    # or blank image 
    else:
        return img


# ## Testing the pipeline on the provided images
# ***
# Here, the pipeline was tested and iterated until a reasonalbe group of parameters
# and function structures was achieved. Parameters included 
# gussian kernel size (15), Canny edge detection low (50) and high (150) thresholds, ROI verticies, though transform parameters
# and image composite mixing parameters for final output
# ***

# In[39]:



# put all together for the pipeline
def detect_lines(img):
    #img1 = select_rgb_white_yellow(img) # source [1]
    img2 = grayscale(img)
    img3 = gaussian_blur(img2, kernel_size = 5)
    img4 = canny(img3, 50, 150)
    
    # define the roi extents
    left_bottom = [0, 539]
    right_bottom = [900, 539]
    apex = [475, 320]
    vertices = np.array([[left_bottom, right_bottom, apex]], dtype=np.int32)
    
    # cut image to roi
    img_roi = region_of_interest(img4, vertices)
    
    # hough line transform, and location averaging and extrapolaiton through y-value assumptions
    input_lines = average_lines(img_roi)
    
    # draw the new lines over the original image
    img6 = final_lane_lines(img, input_lines)
    
    # mix the original and final images
    img7 = weighted_img(img6, img)
    return img7
z=50
# loop through the test images and plot result
for img in test_images:
    #plt.figure()
    plt.tight_layout()
    #global z
    kurs = "output_img/%i.png" % z
    z=z+1
    plt.savefig(kurs, format='png')
    plt.imshow(detect_lines(img))


# ## Testing on Movie files
# 
# ***
# The pipeline was tested on the movie files. Below is the white lines movie.
# ***

# In[40]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(detect_lines) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# In[41]:


HTML("""
<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>
""".format(white_output))


# ## Testing on the white and yellow lines moive 
# 
# ***
# The pipeline performs worse on the yellow and white lines movie, with a few visible 
# missclassifications of roadside pixles as components of lane lines. Overall it still
# identifies lanes well, however, near perfect accuracy is needed for an application 
# like this. Further parameter tuning would improve performance, however, it is unlikey 
# that this particular logic pipeline will generalize well to other road conditions, 
# even if was tuned well to the movie examples given in this project.
# ***
# 

# In[42]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,7)
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(detect_lines)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[43]:


HTML("""
<video width="960" height="540" controls>
  <source src="test_videos_output/solidYellowLeft.mp4">
</video>
""".format(yellow_output))


# In[44]:


#########################DATABASE##############################
sql = "UPDATE `task` SET value = 1 WHERE slno=1;"
number_of_rows = cursor.execute(sql)
db.commit()
db.close()
#########################End-DATABASE###########################

