from sklearn import svm
from sklearn.externals import joblib
import cv2
import numpy as np
import os
import math
np.set_printoptions(threshold=np.inf)

def isvalid(img,i,j):
	if(i<0 or j<0 or i>=img.shape[0] or j>=img.shape[1]):
		return 0
	return 1

def HOG(img):
	winSize=(64,64)
	blockSize=(16,16)
	blockStride=(8,8)
	cellSize=(8,8)
	nbins=9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	return hog.compute(img)

# def ROI(img):
# 	img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 	roi=np.zeros((img.shape),np.uint8)
# 	for i in range(0,img.shape[0]):
# 		for j in range(0,img.shape[1]):
# 			if(img_hsv[i,j,0]>170 or img_hsv[i,j,0]<10):
# 			# if((math.fabs(img[i,j,0]-img[i,j,1])+math.fabs(img[i,j,1]-img[i,j,2])+math.fabs(img[i,j,2]-img[i,j,0]))/100<1):
# 				roi[i,j]=img[i,j]
# 	cv2.imshow('roi',roi)
# 	cv2.waitKey(0)

model=joblib.load('saved_svm.pkl')

test=cv2.imread('./dataset_old/images/00040.ppm')#dataset/03/00011.ppm')
print 'testing'
print test.shape
d = 0
# test=ROI(test)
out=np.zeros((test.shape),np.uint8)
for i in range(0,test.shape[0]):
	for j in range(0,test.shape[1]):
		out[i,j]=test[i,j]
for i in range(0,test.shape[0],10):
	for j in range(0,test.shape[1],10):
		if(isvalid(test,i+64,j+64)==0):
			continue
		if(i%100==0 and j%100==0):
			print i,j
		# patch=np.zeros((64,64,3),dtype=np.uint8)
		patch=test[[k for k in range(i,i+64)],:,:]
		patch=patch[:,[k for k in range(j,j+64)],:]
		# patch=cv2.resize(patch,(64,64))
		if(model.predict(np.reshape(HOG(patch),(-1,1764)))==[1]):
			P1=(j,i)
			P4=(j+64,i+64)
			cv2.rectangle(out,P1,P4,[0,0,255],thickness=1)
			#cv2.imshow("box", out[i:i+64, j:j+64, :]); cv2.waitKey(0);
			# cv2.imwrite("detected"+str(d)+".png", out[i:i+64, j:j+64, :])
			d=d+1
cv2.imshow("out",out)
cv2.waitKey(0)