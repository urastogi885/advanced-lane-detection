import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import argv

cap = cv2.VideoCapture('Night Drive - 2689.mp4')
# video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
video_output = cv2.VideoWriter('outpy3.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 20, (1920, 1080))
total_frames = 0

while True:
    video_frame, frame = cap.read()
    if not video_frame:
        break
    total_frames += 1
    '''kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(frame,-1,kernel)
    plt.imshow(dst)

    nmax = 255 #New maximum
    nmin = 0
    out1 = cv2.normalize(frame,None,alpha = nmin,beta = nmax,norm_type = cv2.NORM_MINMAX)

    alpha =3
    beta = 50
    out = cv2.addWeighted(out1, alpha, np.zeros(frame.shape,frame.dtype),0,beta)'''

    average += frame
    average /= len(files)

    # cv2.imshow('app',frame)
    cv2.imshow('app2', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_output.write(out)

cap.release()
video_output.release()
cv2.destroyAllWindows()