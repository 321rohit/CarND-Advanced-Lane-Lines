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
ref_image:./image_output/cameracalibration

ref_image:./image_output/image_pipeline

Camera Calibration
1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
###ref_image:./image_output/cameracalibration.jpg

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()`function.
 Some of the chessboard images don't appear because findChessboardCorners was unable to detect the desired number of internal corners.

Pipeline (single images)


1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
###ref_image:./image_output/undistorrted_output



2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.
ref_image:./image_output/colorgradient
I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step.

I have used many gradient thresholds for eg.
a)sobel absolute threshold
b)sobel magnintude threshold
c)sobel direction threshold 
and also the combination of the 
d)sobel magnitude and direction


and i have tried various colorspaces channels 
a)RGB colorspsace having R,G,B channels
b)HLS colorspace having H,L,S channels
c)HSV colorspace having H,S,V channels 
The HSV color space has the following three components

H – Hue ( Dominant Wavelength ).
S – Saturation ( Purity / shades of the color ).
V – Value ( Intensity ).

d)LAB coorspace
The Lab color space has three components.

L – Lightness ( Intensity ).
a – color component ranging from Green to Magenta.
b – color component ranging from Blue to Yellow

https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
In the end I combined a threshold on the B channel of the Lab colorspace to extract yellow lines and a threshold 
on the L channel of the HLS colorspace to extract the whites.

3. Describe how  you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform is titled "Perspective Transform" in the Jupyter notebook, in the seventh and eighth code cells from the top. The unwarp() function takes as inputs an image (img), as well as source (src) and destination (dst) points. I chose to hardcode the source and destination points in the following manner:

src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]]])
dst = np.float32(
    [[(img_size[0] / 4)+80, 0],
    [(img_size[0] * 3 / 4), 0],
    [(img_size[0] / 4)+80, img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]]])
    
    
| Source        | Destination   |
|:-------------:|:-------------:| 
| 585, 460      | 400, 0        | 
| 700, 460      | 960, 0        |
| 250, 675      | 400, 720      |
| 1100com,675      | 960, 720      |

4. Describe how you identified lane-line pixels and fit their positions with a polynomial?
I have created a function sliding_window_polyfit for this.First of all i computes histogram of the bottom half of the image
to find the starting position for the left and right lanes.Histogram peaks gives us thestarting points for the  laeft and right lanes.
In the lecture notes,we had midpoint to ,where left to midpoint was left lane and right to it was right lane but i add a quarter_point also
which is a histogram of quareters so that we had left and right linees justnext to mmidpoint.
This helps to reject the adjacent lanes.The function then identifies ten windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image(nonzero).Now we identified the pixels belonging to right and left lines,just fit them to a second order polynomial to each set of pixels.
and polyfit_using_prev_fit funtion using the previous state of the frame to determine in a video.The previous left and right fits and the currents left and right fits helps to draw these lines.
ref_image:./image_output/polyfit_line

5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
ref_image:./image_output/radius_curve
The formula for the calculation of raddius of curvature is calculated as:
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
or more generally :
        curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])

In this example, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. y_0 is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_meters_per_pixel is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

position of the vehicle with respect to center:
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
r_fit_x_int and l_fit_x_int are the x-intercepts of the right and left fits, respectively. This requires evaluating the fit at the maximum y value (719, in this case - the bottom of the image) because the minimum y value is actually at the top (otherwise, the constant coefficient of each fit would have sufficed). The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I have implemented this step in the function named as draw_lane and draw_info respectively.
ref_image:./image_output/radius_curve

Pipeline (video)
 Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
 I have saved my output video in the folder named as project_video_output.mp4

### Discussion
 Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
a)I have faced many problems while implementing this project.As checkpoints were not given,this makes even more harder to implement this 
project.but lecture notes helps a lot to overcome as they give very much details.
b)A lot of research was done to get the proper color channelsand get out  info out of them.
c)in implementing the sliding_window_polyfit,i have faced the issues in drawing the rectangles.
d)what need to be done to define a class to store the characteristics.
e)upto perspective transform it was not difficult to implement,but after that i need to go very deep to understand the things,atleast it was the case for me.
f)my pipeline could be failed when there will be different sizes of lanes would be there,somewhere large somewhere smmall



