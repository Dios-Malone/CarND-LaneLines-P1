# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
* First, I converted the images to grayscale. The image was copied so that I have an original image as reference. 
* Then I applied Gaussian blur on the grayscale image with size of 5. I understood that Canny function would do Gaussian bluring implicitly. 
* Aterwards, I detected the edges from the blurred image by using Canny function. 
* And I masked the edges and only kept the interested region. 
* Then I applied Hough transform to find out the linear lines from the interested region and created a layer image with detected lines drawn.
* Finally, I added the layer image with detected lines onto original image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by categorizing lines for left lane and right lane by its gradient. The lines out of the fesible gradient range were filter out. Afterwards, I created one line for left lane and one line for right line by using the average value from the left and right lines group.


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be that there are no lines found for either left lane or right lane when the road is turning in curve. The lines gradient will be out of the pre-defined range.
Another shortcoming could be that the pre-defined region of interest would not be valid anymore if the position of camera changes.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to change the gradient range from hard coded values to dynamic values. The gradient range could be adjust according to the previous frame's line gradients. The rate of line gradient change should not be too fast. For example, I could define the gradient range as previous frame's gradient +/- 5%

Another potential improvement could be to define multiple region of interest, one for the close area and one for the far area. And use the linear line model to dectect lines from close area and use curve line model to detect lines from far area. By this way, it could hanlde the turning road better.
