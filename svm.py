from sklearn import svm
from sklearn.externals import joblib
import cv2
import numpy as np
import os


def images_and_hog(folder):
	X=[]
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
	count=0
	for file in os.listdir(folder):
		img=cv2.imread(os.path.join(folder,file))
		if img is not None:
			count+=1
			img=cv2.resize(img,(64,64))
			hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
			feat=hog.compute(img)
			X.append(feat)
	X=np.array(X)
	X=np.reshape(X,(count,1764))
	return X

X=np.array([])
for i in range (0,42):
	folder='./dataset/0'+str(i) if i<10 else './dataset/'+str(i)
	X=np.append(X,images_and_hog(folder))

X=np.reshape(X,(-1,1764))
Y=np.array([1 for i in range(0,X.shape[0])])

