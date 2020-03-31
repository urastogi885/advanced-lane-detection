from glob import glob
import numpy as np
import cv2 as cv
import copy

vidname1 = 'data_1/data/*.png'  # Input challenge video file
images_video = 'data_1/data1_output.avi'  # Input project video file
vidname2 = 'data_1/image_dataset_output.avi'


video_format = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
video_input = cv.VideoWriter(images_video, video_format, 10.0, (1392, 512))
video_output = cv.VideoWriter(vidname2, video_format, 10.0, (1281, 372))

# Calibration Matrix
calibration_matrix = np.array(
    [[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02],
     [0.000000e+00, 0.000000e+00, 1.000000e+00]])
# Distortion Coefficients
distortion = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

# Parameters for changing Contrast
array_alpha = np.array([1.25])
# Parameters for changing Brightness
array_beta = np.array([-100.0])

# Lower and upper bounds for masking of yellow color in HSV color space
lower_yellow = np.array([5, 140, 70])
upper_yellow = np.array([70, 255, 255])

# Lower and upper bounds for masking of white color in BGR color space
lower_white = np.array([230, 230, 230])
upper_white = np.array([255, 255, 255])

tempxyellow = []
tempyyellow = []
tempxwhite = []
tempywhite = []

total_frames = 0
direction = ''
turn_count = {'Straight': 0, 'Turning Left': 0, 'Turning Right': 0}

# images_array = []
for file in glob(vidname1):
    frame = cv.imread(file)
    # print(frame.shape)
    video_input.write(frame)
video_input.release()

cap = cv.VideoCapture(images_video)  # Reading the video file
while True:
    ret, video_frame = cap.read()
    # If no video frame is generated or the video has ended
    if not ret:
        print('break')
        break
    total_frames += 1
    h, w, _ = video_frame.shape  # Getting height and width of the frame
    # Refining camera calibration parameters for undistortion
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibration_matrix, distortion, (w, h), 1, (w, h))
    dst = cv.undistort(video_frame, calibration_matrix, distortion, None, newcameramtx)  # Undistorting the image
    x, y, w, h = roi  # parameters for reconverting the undistort frame into normal sqauare frame
    undistort = dst[y:y + h, x:x + w]  # Undistorted frame
    undistort_copy = undistort.copy()
    undistort = cv.fastNlMeansDenoisingColored(undistort, None, 10, 10, 7, 21)
    # Four corners points in the camera frame for homography
    src = np.float32([[int((w / 4) + 80), int((h / 2) + 100)], [int((w / 2) + 100), int((h / 2) + 100)],
                      [int((w / 2) + 180), h], [int((w / 4) - 80), h]])
    # Four corners points in the world frame for homography
    snk = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32)
    # Homography matrix using the above four points in camera frame and world frame.
    H = cv.getPerspectiveTransform(src, snk)
    Hinv = np.linalg.inv(H)  # Getting Inverse of the homography matrix
    # Using wrap Perspective to get the required area of interest in front of us
    homoimage = cv.warpPerspective(undistort, H, (200, 200))
    mask2 = cv.inRange(homoimage, lower_white, upper_white)  # Masking the white color alone from the BGR color space
    # Finding corners using Shi-Tomasi method in the mask of the white color
    cornerswhite2 = cv.goodFeaturesToTrack(mask2, 100, 0.01, 0.05)
    xyellow = []  # Variable for storing the x-coordinates of the corners of the yellow color
    yyellow = []  # Variable for storing the y-coordinates of the corners of the yellow color
    xwhite = []  # Variable for storing the x-coordinates of the corners of the white color
    ywhite = []  # Variable for storing the y-coordinates of the corners of the white color

    try:  # When corners are found
        for i in cornerswhite2:  # Looping over the corners detected from the white color
            x, y = i.ravel()  # Separating x and y coordinates
            if x > 170:
                xwhite.append(x)  # Appending x coordiante in the xwhite list
                ywhite.append(y)  # Appending y coordiante in the xwhite list
            if x < 40:
                xyellow.append(x)  # Appending x coordiante in the xyellow list
                yyellow.append(y)  # Appending y coordiante in the yyellow list
    except:  # When no corners are found (few frames in the challenge video)
        xyellow = copy.deepcopy(tempxyellow)  # Using last frame's x-coordiante data
        yyellow = copy.deepcopy(tempyyellow)  # Using last frame's y-coordinate data
        xwhite = copy.deepcopy(tempxwhite)  # Using last frame's x-coordiante data
        ywhite = copy.deepcopy(tempywhite)  # Using last frame's y-coordinate data
    if len(xyellow) == 0 or len(yyellow) == 0 or len(xwhite) == 0 or len(ywhite) == 0:
        xyellow = copy.deepcopy(tempxyellow)  # Using last frame's x-coordiante data
        yyellow = copy.deepcopy(tempyyellow)  # Using last frame's y-coordinate data
        xwhite = copy.deepcopy(tempxwhite)  # Using last frame's x-coordiante data
        ywhite = copy.deepcopy(tempywhite)  # Using last frame's y-coordinate data

    zyellow = np.polyfit(yyellow, xyellow, 1)  # Polyfitting a line in corners corresponding to the yellow lane
    fyellow = np.poly1d(zyellow)  # Equation of the line polyfitted in the yellow lane

    zwhite = np.polyfit(ywhite, xwhite, 1)  # Polyfitting a line in corners corresponding to the white lane
    fwhite = np.poly1d(zwhite)  # Equation of the line polyfitted in the white lane

    yplotyellow = np.array([20, 200])  # Taking two x-coordiante for plotting the yellow line
    xplotyellow = fyellow(yplotyellow)  # Calculating the corresponding y-coordiante for plotting the yellow line

    yplotwhite = np.array([20, 200])  # Taking two x-coordiante for plotting the white line
    xplotwhite = fwhite(yplotwhite)  # Calculating the corresponding y-coordiante for plotting the white line
    # Calculating length from the middle of the frame for determining turning
    length1 = int(xplotyellow[0] + (xplotwhite[0] - xplotyellow[0]) / 2)

    x1, y1, z1 = np.matmul(Hinv, [xplotyellow[0], yplotyellow[0],
                                  1])  # Calculating the point 1 at yellow lane in the camera frame using inverse homography
    x2, y2, z2 = np.matmul(Hinv, [xplotyellow[1], yplotyellow[1],
                                  1])  # Calculating the point 2 at yellow lane the camera frame using inverse homography

    x3, y3, z3 = np.matmul(Hinv, [xplotwhite[0], yplotwhite[0],
                                  1])  # Calculating the point 1 at white lane in the camera frame using inverse homography
    x4, y4, z4 = np.matmul(Hinv, [xplotwhite[1], yplotwhite[1],
                                  1])  # Calculating the point 2 at white lane in the camera frame using inverse homography
    x5, y5, z5 = np.matmul(Hinv, [85, 150, 1])

    cv.line(undistort, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 0, 255),
            12)  # Drawing line at yellow lane
    cv.line(undistort, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 0, 255),
            12)  # Drawing line at white lane
    pts = np.array([[(int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)),
                     (int(x3 / z3), int(y3 / z3))]], np.int32)

    cv.fillPoly(undistort, pts, (128, 255, 128))  # Filling the polygon with the green color
    if length1 < 93:  # If the length1 is less than 93 then the car is turning left
        turn_count['Turning Left'] += 1
    elif length1 > 107:  # If the length1 is greater than 107 then the car is turning left
        turn_count['Turning Right'] += 1
    elif 95 < length1 < 107:  # If the length1 is less than 93 and greater than 107 then the car is moving straight
        turn_count['Straight'] += 1

    if total_frames % 8 == 0:
        direction = max(turn_count, key=turn_count.get)
        turn_count = turn_count.fromkeys(turn_count, 0)

    cv.putText(undistort, direction, (int(x5 / z5), int(y5 / z5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    tempxyellow = copy.deepcopy(xyellow)  # Storing current frame x-coordiantes for yellow color
    tempyyellow = copy.deepcopy(yyellow)  # Storing current frame y-coordiantes for yellow color
    tempxwhite = copy.deepcopy(xwhite)  # Storing current frame x-coordiantes for white color
    tempywhite = copy.deepcopy(ywhite)  # Storing current frame y-coordiantes for white color

    cv.imshow("Lane_Detection", undistort)  # Output of the detected lanes in the input video
    cv.imshow("Homo", homoimage)  # Output of the detected lanes in the input video
    key = cv.waitKey(1)
    if key == 27:
        break
    video_output.write(undistort)
    # print(undistort.shape)
video_output.release()
cv.destroyAllWindows()
# end
