import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from PIL import Image, ImageFilter



cap = cv2.VideoCapture('Night Drive - 2689.mp4')
#video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
video_output = cv2.VideoWriter('prob1_clahe + convo.avi',cv2.VideoWriter_fourcc('X','V','I','D'), 20,(1920,1080))
total_frames = 0

def gamma(img,g=1.00):
    invGamma = 1.0 / g
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

while True:
    video_frame, frame = cap.read()
    if not video_frame:
        break
    total_frames += 1
    
    # Convolution
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(frame,-1,kernel)
    plt.imshow(dst)
    
    # CLAHE Method applied
    
    lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Gamma Correction
    '''
    gamma_img = gamma(frame,2.2)
    out = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    '''
    # max min linear contrast stretch
    '''
    nmax = 255 #New maximum
    nmin = 0
    out = cv2.normalize(frame,None,alpha = nmin,beta = nmax,norm_type = cv2.NORM_MINMAX)
    '''
    # Alpha_beta histogram method
    '''
    alpha =3
    beta = 50
    out = cv2.addWeighted(frame, alpha, np.zeros(frame.shape,frame.dtype),0,beta)
    '''

    # Mean Filtering
    '''
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting to HSV
    figure_size = 9 # defining the size of the kernel for x and y
    mean_filter = cv2.blur(image,(figure_size, figure_size))
    plt.figure(figsize=(11,6))
    out = cv2.cvtColor(mean_filter, cv2.COLOR_HSV2RGB)
    '''
    
    # Median Filtering
    '''
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    figure_size = 9 # defining the size of the kernel for x and y
    median_filter = cv2.medianBlur(image, figure_size)
    out = cv2.cvtColor(median_filter, cv2.COLOR_HSV2RGB)
    '''
    
    #cv2.imshow('app',frame)
    cv2.imshow('app2',out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_output.write(out)
    

    
cap.release()
video_output.release()
cv2.destroyAllWindows()

