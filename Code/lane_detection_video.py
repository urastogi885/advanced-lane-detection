import cv2
import copy
import numpy as np
from sys import argv
from utils import constants

script, video_location, output_destination = argv

cap = cv2.VideoCapture(str(video_location))  # Reading the video file
video_format = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
video_output = cv2.VideoWriter(str(output_destination), video_format, 30.0, (1200, 617))

temp_x_yellow = []
temp_y_yellow = []
temp_x_white = []
temp_y_white = []

while True:
    ret, video_frame = cap.read()
    # If no video frame is generated or the video has ended
    if not ret:
        break
    constants.total_frames += 1
    # Get height and width of the video frame
    h, w = video_frame.shape[:2]
    # Get optimal camera calibration parameters to undistort
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(constants.data2_camera_matrix,
                                                           constants.data2_distortion_matrix,
                                                           (w, h), 1, (w, h))
    # Undistort the image
    undistorted_img = cv2.undistort(video_frame, constants.data2_camera_matrix, constants.data2_distortion_matrix, None,
                                    new_camera_matrix)
    # Remove curved parts gained after undistortion
    x, y, w, h = roi
    undistort = undistorted_img[y:y + h, x:x + w]
    undistort_copy = undistort.copy()
    undistort = cv2.fastNlMeansDenoisingColored(undistort, None, 10, 10, 7, 21)
    # Four corners points in the camera frame for homography
    section_points = np.float32([[int((w / 2) - 80), int((h / 2) + 100)],
                                 [int((w / 2) + 120), int((h / 2) + 100)],
                                 [int(w - 90), int(h - 30)],
                                 [80, h]])
    # Homography matrix using the above four points in camera frame and world frame.
    H = cv2.getPerspectiveTransform(section_points, constants.warp_points)
    Hinv = np.linalg.inv(H)  # Getting Inverse of the homography matrix
    homo_image = cv2.warpPerspective(undistort, H, (constants.warp_size[0], constants.warp_size[1]))
    # Make a copy of the image recieved through homography
    homo_image_copy = copy.deepcopy(homo_image)
    cv2.multiply(homo_image_copy, constants.array_alpha, homo_image)
    cv2.add(homo_image_copy, constants.array_beta, homo_image_copy)
    cv2.multiply(homo_image_copy, constants.array_alpha, homo_image_copy)
    # Convert the above into HSV color space for yellow color detection
    hsv = cv2.cvtColor(homo_image_copy, cv2.COLOR_BGR2HSV)
    # Mask the yellow color alone from the HSV color space
    mask1 = cv2.inRange(hsv, constants.yellow_lower, constants.yellow_upper)
    # Mask the white color alone from the BGR color space
    mask2 = cv2.inRange(homo_image, constants.white_lower, constants.white_upper)
    corners_yellow = cv2.goodFeaturesToTrack(mask1, 300, 0.01, 0.05)
    corners_white = cv2.goodFeaturesToTrack(mask2, 100, 0.01, 0.05)

    xyellow = []  # Variable for storing the x-coordinates of the corners of the yellow color
    yyellow = []  # Variable for storing the y-coordinates of the corners of the yellow color
    xwhite = []  # Variable for storing the x-coordinates of the corners of the white color
    ywhite = []  # Variable for storing the y-coordinates of the corners of the white color
    # When corners are found
    try:
        for i in corners_white:  # Looping over the corners detected from the white color
            x, y = i.ravel()  # Separating x and y coordinates
            if x > 130:
                # Append coordinates in the white list
                xwhite.append(x)
                ywhite.append(y)

        for i in corners_yellow:  # Looping over the corners detected from the yellow color
            x, y = i.ravel()  # Separating x and y coordinates
            if x < 70:
                # Append coordinates in the yellow list
                xyellow.append(x)
                yyellow.append(y)
    except:
        # When no corners are found (few frames in the challenge video)
        xyellow = temp_x_yellow[:]  # Using last frame's x-coordiante data
        yyellow = temp_y_yellow[:]  # Using last frame's y-coordinate data
        xwhite = temp_x_white[:]  # Using last frame's x-coordiante data
        ywhite = temp_y_white[:]  # Using last frame's y-coordinate data
    # Poly-fit a line in corners corresponding to the yellow lane
    zyellow = np.polyfit(yyellow, xyellow, 1)
    # Equation of the line polyfitted in the yellow lane
    fyellow = np.poly1d(zyellow)
    # Poly-fit a line in corners corresponding to the white lane
    zwhite = np.polyfit(ywhite, xwhite, 1)
    # Equation of the line polyfitted in the white lane
    fwhite = np.poly1d(zwhite)
    # Take two x-coordiante for plotting the yellow line
    yplotyellow = np.array([20, 197])
    # Calculate the corresponding y-coordiante for plotting the yellow line
    xplotyellow = fyellow(yplotyellow)
    # Take two x-coordiante for plotting the white line
    yplotwhite = np.array([20, constants.warp_size[1]])
    # Calculate the corresponding y-coordiante for plotting the white line
    xplotwhite = fwhite(yplotwhite)
    # Calculate length from the middle of the frame for determining turning
    length1 = int(xplotyellow[0] + (xplotwhite[0] - xplotyellow[0]) / 2)
    # Calculate the point 1 at yellow lane in the camera frame using inverse homography
    x1, y1, z1 = np.matmul(Hinv, [xplotyellow[0], yplotyellow[0], 1])
    # Calculate the point 2 at yellow lane the camera frame using inverse homography
    x2, y2, z2 = np.matmul(Hinv, [xplotyellow[1], yplotyellow[1], 1])
    # Calculate the point 1 at white lane in the camera frame using inverse homography
    x3, y3, z3 = np.matmul(Hinv, [xplotwhite[0], yplotwhite[0], 1])
    # Calculate the point 2 at white lane in the camera frame using inverse homography
    x4, y4, z4 = np.matmul(Hinv, [xplotwhite[1], yplotwhite[1], 1])
    x5, y5, z5 = np.matmul(Hinv, [85, 150, 1])
    # Draw line at yellow line
    cv2.line(undistort, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 0, 255), 12)
    # Draw line at white line
    cv2.line(undistort, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 0, 255), 12)
    points = np.array([[(int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)),
                     (int(x3 / z3), int(y3 / z3))]], np.int32)
    # Fill the polygon with the green color
    cv2.fillPoly(undistort, points, (128, 255, 128))
    # Check for direction of turning
    if length1 < 93:
        constants.turn_count['Turning Left'] += 1
    elif length1 > 107:
        constants.turn_count['Turning Right'] += 1
    elif 95 < length1 < 107:
        constants.turn_count['Straight'] += 1
    # Get maximum of directions deduced over 8 frames to avoid noisy predictions
    if constants.total_frames % 8 == 0:
        direction = max(constants.turn_count, key=constants.turn_count.get)
        constants.turn_count = constants.turn_count.fromkeys(constants.turn_count, 0)
    # Display direction on output frame
    cv2.putText(undistort, constants.direction, (int(x5 / z5), int(y5 / z5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # Store previous corners
    temp_x_yellow = xyellow[:]
    temp_y_yellow = yyellow[:]
    temp_x_white = xwhite[:]
    temp_y_white = ywhite[:]
    # Append final output to video file
    video_output.write(undistort)
video_output.release()
cap.release()
cv2.destroyAllWindows()
