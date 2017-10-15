from sklearn import svm
from sklearn.externals import joblib
import cv2
import numpy as np
import os
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


def images_and_hog(folder):
	X=[]
	X_test=[]
	count=0
	count_test=0
	for file in os.listdir(folder):
		img=cv2.imread(os.path.join(folder,file))
		if img is not None:
			img=cv2.resize(img,(64,64))
			feat=HOG(img)
			count+=1
			if(count%10==0):
				X_test.append(feat)
				count_test+=1
			else:
				X.append(feat)
	X=np.array(X)
	X=np.reshape(X,(count-count_test,1764))
	X_test=np.reshape(X_test,(count_test,1764))
	return X,X_test

#**************Training***********

X=np.array([])
X_test=np.array([])

for i in range (0,42):
	folder='./dataset/0000'+str(i) if i<10 else './dataset/000'+str(i)
	x,x_test=images_and_hog(folder)
	X=np.append(	X,x)
	X_test=np.append(X_test,x_test)
X=np.reshape(X,(-1,1764))
X_test=np.reshape(X_test,(-1,1764))
print "Shape=",X.shape,X_test.shape
Y=np.array([1 for i in range(0,X.shape[0])])
num_positive_example=X.shape[0]
num_positive_test=X_test.shape[0]

print "Positive exm ",num_positive_example
folder='./neg_exm/'#'./dataset/non_signs/'
x,x_test=images_and_hog(folder)
X=np.append(X,x)
X_test=np.append(X_test,x_test)
X=np.reshape(X,(-1,1764))
X_test=np.reshape(X_test,(-1,1764))
print X.shape[0]
Y=np.append(Y,[0 for i in range(0,X.shape[0]-num_positive_example)])


print "shape =",X.shape,Y.shape
model=svm.SVC(C=100, cache_size=5000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model.fit(X,Y)
model.score(X,Y)
joblib.dump(model,'saved_svm.pkl')
# model=joblib.load('saved_svm.pkl')


predict=model.predict(X_test)
corr=0
error=0
for i in range(0,predict.shape[0]):
	if(i<num_positive_test):
		if(predict[i]==1):
			corr+=1
		else: error+=1
	else:
		if(predict[i]==0):
			corr+=1
		else:
			error+=1

print("Accuracy : ",float((corr*100.0)/(corr+error)))
print corr,error


# test=cv2.imread('./dataset/00040.ppm')#dataset/03/00011.ppm')
# print 'testing'
# print test.shape
# out=np.zeros((test.shape),np.uint8)
# for i in range(0,test.shape[0]):
# 	for j in range(0,test.shape[1]):
# 		out[i,j]=test[i,j]
# for i in range(0,test.shape[0],10):
# 	for j in range(0,test.shape[1],10):
# 		if(isvalid(test,i+64,j+64)==0):
# 			continue
# 		if(i%100==0 and j%100==0):
# 			print i,j
# 		# patch=np.zeros((64,64,3),dtype=np.uint8)
# 		patch=test[[k for k in range(i,i+64)],:,:]
# 		patch=patch[:,[k for k in range(j,j+64)],:]
# 		# patch=cv2.resize(patch,(64,64))
# 		if(model.predict(np.reshape(HOG(patch),(-1,1764)))==[1]):
# 			P1=(j,i)
# 			P4=(j+64,i+64)
# 			cv2.rectangle(out,P1,P4,[0,0,255],thickness=1)
# cv2.imshow("out",out)
# cv2.waitKey(0)
