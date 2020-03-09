import cv2
import numpy as np


def find_roi(img):
    """
    Find a region of interest within an image
    :param img: opencv image to find the ROI
    :return: image with ROI highlighted
    """
    # Define a mask of the same size as the image
    mask = np.zeros_like(img)
    # Get vertices for the region of interest
    shape = img.shape
    lower_left = [0, shape[0] - 20]
    lower_right = [shape[1], shape[0] - 20]
    top_left = [0, shape[0] / 2 + 20]
    top_right = [shape[1], shape[0] / 2 + 20]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    # print(vertices)
    # Define mask color based on number of channels in the image
    if len(shape) > 2:
        mask_color = (255,) * shape[2]
    else:
        mask_color = 255

    # Fill pixels inside the polygon defined by vertices
    cv2.fillPoly(mask, vertices, mask_color)

    # Return an image where only mask pixels are non-zero
    return cv2.bitwise_and(img, mask)


def get_line_coordinates(lines):
    rho_left = []
    rho_right = []
    theta_left = []
    theta_right = []
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if np.pi * 0.2 < theta < np.pi * 0.4:
                    rho_left.append(rho)
                    theta_left.append(theta)
                elif np.pi * 0.6 < theta < np.pi * 0.8:
                    rho_right.append(rho)
                    theta_right.append(theta)
    # Do not want the lines with maximum votes
    left_rho = np.median(rho_left)
    right_rho = np.median(rho_right)
    left_theta = np.median(theta_left)
    right_theta = np.median(theta_right)


if __name__ == '__main__':
    camera_matrix = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                              [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # [-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]
    distortion_matrix = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
    lane_points = np.float32([[493, 300], [729, 300], [860, 480], [36, 480]])
    height = 820
    width = 640
    final_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    # img_location = 'data_1/data/0000000221.png'
    video = cv2.VideoCapture('data_2/challenge_video.mp4')
    # print(img_location)
    while True:
        img_frame_exists, img_frame = video.read()
        if not img_frame_exists:
            break
        # img_frame = cv2.imread(img_location)
        img_width, img_height, img_channels = img_frame.shape
        h_matrix, ret_val = cv2.findHomography(lane_points, final_points)
        # warped_img = cv2.warpPerspective(img_frame, mask, (width, height))
        # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_matrix,
        #                                                      (img_width, img_height), 1, (img_width, img_height))
        # x, y, w, h = roi
        undistorted_img = cv2.undistort(img_frame, camera_matrix, distortion_matrix)  # , newCameraMatrix=new_camera_matrix)
        denoised_img = cv2.fastNlMeansDenoisingColored(undistorted_img, None, 10, 10, 7, 21)
        edges = cv2.Canny(denoised_img, 50, 150)
        # edges = edges[155:425, 200:730]
        roi_img = find_roi(edges)
        hough_lines = cv2.HoughLines(roi_img, 2, np.pi / 180, 30)  # , np.array([]), maxLineGap=200, minLineLength=50)

        hough_img = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), dtype=np.uint8)
        # for rho, theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * a)
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * a)
        #
        #     cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # hough_img = np.zeros((edges.shape[0], edges.shape[1]))
        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        final = cv2.addWeighted(img_frame, 0.8, hough_img, 1, 0)

        cv2.imshow("Original", img_frame)
        # cv2.imshow("Warp", warped_img)
        cv2.imshow("Undistorted", undistorted_img)
        cv2.imshow("Denoised", denoised_img)
        cv2.imshow("Edges", edges)
        cv2.imshow("ROI", roi_img)
        # cv2.imshow("warp", hough_img)
        # cv2.imshow("final", final)
        # cv2.imshow("Undistorted", undistorted_if)
        key = cv2.waitKey(0)
        # if key == 27:
        #     break
