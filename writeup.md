## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell of the IPython notebook under header `Camera calibration and undistortion` located in "./Project2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_points_batch` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To apply distortion correction to an image I use `undistort(img, state)` function that serves as a wrapper for `cv2.undistort(...)`. `state` here is an instance of `State` class that I created to store various values that are useful for the pipeline. 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The transformations are applied to an already warped image, because that worked better on the images that I tested on. I implemented this in 4 steps:
a. `get_binary_image_with_edge_pixels(img, state)`
This function uses `cv2.Sobel()`, normalizes the result to fit into [0..1] range, using a constant `MAX_EDGE_VALUE`, which is a theoretical max value for `cv2.Sobel()` output, and then thresholds the result using `THRESHOLD_EDGE = 0.05`. The output of this function is a binary image.
b. `get_white_line_signals(img, state)`
There's some light-correction in this function - we know that in the test videos there are parts of the roads that are much brighter than most of the road. To take this into account I convert the image to HLS to get the light matrix, take a central vertical strip of it(to avoid sides of the road), then take a 98-percentile along Y-axis to avoid white lines and the smoothen that vector by convolving it. Then I use that vector to build `convolved_matrix` that consists of just replicated vectors. Finally, I use this expression `binary[light_matrix > convolved_matrix + 20] = 1` to get the binary image. At first I tried dividing `light_matrix` by `convolved_matrix`, but this expression worked best for me.
c. `get_binary_image_with_yellow_pixels(img, state)`
This is a simple thresholding of an HLS image using the Hue channel.
d. `get_combined_line_pixels(img, state)`
This function calls the 3 functions above to get 3 binary images. White and yellow pixels are combined with `np.bitwise_or(white, yellow)`. Then I do a 2D convolution for edge pixels and for color pixels separately to kind of spread the signals to increase the area of intersection between colors and edges. Finally, I combine colors and edges using `np.sqrt(edges_sum * colors_sum)` and then normalize the matrix. As you can tell the result is a 2D matrix, but it's not binary - it kind of contains the strength of the signal.

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp(img, state)`, which appears in code cell under title `Warping` in the file `Project2.ipynb`. 
The function uses a precalculated `warp_matrix` that is stored in `state`.  I chose the hardcode the source and destination points in the following manner:

```python
# destination
WARPED_POINTS = np.array([
    [WARPED_WIDTH * 0, WARPED_HEIGHT - 1],
    [WARPED_WIDTH * 0, 0],
    [WARPED_WIDTH - 1, 0],
    [WARPED_WIDTH - 1, WARPED_HEIGHT - 1],
], dtype=np.float32)

# source
def build_region_of_interest(shape):
    height = shape[0]
    width = shape[1]
    return np.array([[
        [width * 0.0, height - 1],
        [width * 0.415, height * 0.65],
        [width * 0.585, height * 0.65],
        [width - 1, height - 1],
    ]], dtype=np.int32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

At this step the input is a 2D matrix that stores combined edge&color signals. First, we need to find windows on an image. If we don't have a curve yet from previous frames, we look for 2 peaks(right/left) in signals in the bottom part of the matrix and place the bottom windows there and then look for the rest of windows based on the bottom windows. If we have a curve from previous frames - we look for new windows along the curve. The X-position of new windows is calculated by the function `find_peak(img_patch, conv_kernel)`, which sums up values along Y-axis and convolves them to find a peak location. You can also see some calculations involving `inertia_...` variables there - this is used to add some friction to make it more likely that the peak will be found closer to the center of the `img_patch` and less likely that the windows will jump around all the time.
When the windows are found, the `get_curve(windows_for_side, side, state)` function goes over them and builds arrays `points_x` and `points_y`  which are then used to call `np.polyfit(...)`. The input matrix is not binary, so the value at x=45, y=89 can be e.g. `5.3`. In that case, these coordinates will be added to `points_x` and `points_y` 5 times to give the point more weight.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated by the `get_curve_radius()` function in `Curve` class. It uses hardcoded constants for pixels to meters conversion. 
Position of the vehicle is calculated in function `get_stats(state)` which uses already calculated curves to get the deviation.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

When processing a video I use some sort of a weighted average of all previous edge&color frames. The formula that I use is:
`averaged_pixels = image * 0.15 + averaged_pixels * 0.85`
That helps to make line detection more resilient, especially with dashed lines.

I also found it more useful for myself to output not just the result image but also results of other steps of the pipeline, so every frame in my output video contains:
1. warped image with lines
2. edges, white line pixels and yellow line pixels, drawn with different colors on one image
3. combined edges&colors image
4. image with windows and lines on top of combined edges&colors
5. unwarped image with lane
6. curvature and car position 

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
