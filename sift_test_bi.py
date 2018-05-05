import cv2
import numpy as np
import sys
filepath = sys.argv[1];
img = cv2.imread(filepath)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
print("Number of kps detected: {}".format(len(kp)))
img=cv2.drawKeypoints(gray,kp,img)
cv2.imshow('SIFT_built_in',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
