from numpy import array, float32

# Global constants
# Parameters for changing Contrast
array_alpha = array([1.25])
# Parameters for changing Brightness
array_beta = array([-100.0])
# Thresholds from yellow color in HSV color space
yellow_lower = array([5, 140, 70])
yellow_upper = array([70, 255, 255])
# Threshold white color in BGR color space
white_lower = array([230, 230, 230])
white_upper = array([255, 255, 255])

total_frames = 0
direction = ''
turn_count = {'Straight': 0, 'Turning Left': 0, 'Turning Right': 0}

# Constants for dataset 2
data2_camera_matrix = array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                            [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
data2_distortion_matrix = array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
warp_size = (200, 200)
# Four corners points in the homography frame
warp_points = float32([[0, 0], [warp_size[0], 0], [warp_size[0], warp_size[1]], [0, warp_size[1]]])
