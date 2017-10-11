from sklearn import svm
from sklearn.externals import joblib
import cv2
import numpy as np
import os

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


def images_and_hog(folder):
	X=[]
	count=0
	for file in os.listdir(folder):
		img=cv2.imread(os.path.join(folder,file))
		if img is not None:
			count+=1
			img=cv2.resize(img,(64,64))
			feat=HOG(img)
			X.append(feat)
	X=np.array(X)
	X=np.reshape(X,(count,1764))
	return X

# X=np.array([])
# for i in range (0,42):
# 	folder='./dataset/0'+str(i) if i<10 else './dataset/'+str(i)
# 	X=np.append(X,images_and_hog(folder))
# X=np.reshape(X,(-1,1764))
# Y=np.array([1 for i in range(0,X.shape[0])])
# num_positive_example=X.shape[0]

# print num_positive_example
# folder='./neg_exm/'#'./dataset/non_signs/'
# X=np.append(X,images_and_hog(folder))
# X=np.reshape(X,(-1,1764))
# print X.shape[0]
# Y=np.append(Y,[0 for i in range(0,X.shape[0]-num_positive_example)])


# print X.shape,Y.shape
model=svm.SVC(C=5, cache_size=5000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
# model.fit(X,Y)
# model.score(X,Y)
# joblib.dump(model,'saved_svm.pkl')
model=joblib.load('saved_svm.pkl')


test=cv2.imread('./dataset/00135.ppm')#dataset/03/00011.ppm')
# p1=(0,0)
# p2=(64,64)
# cv2.rectangle(test,p1,p2,[0,0,255],thickness=2)
print 'testing'
print test.shape
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
cv2.imshow("out",out)
cv2.waitKey(0)

# predict=model.predict(np.reshape(HOG(test),(-1,1764)))
# print predict