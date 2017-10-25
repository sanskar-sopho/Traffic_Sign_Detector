import cv2
import numpy as np
import tensorflow as tf
import os
np.set_printoptions(threshold=20)


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
	folder='./dataset_old/0'+str(i) if i<10 else './dataset_old/'+str(i)
	x,x_test=images_and_hog(folder)
	X=np.append(X,x)
	X_test=np.append(X_test,x_test)
X=np.reshape(X,(-1,1764))
X_test=np.reshape(X_test,(-1,1764))
print "Shape=",X.shape,X_test.shape
Y=np.array([1 for i in range(0,X.shape[0])])
num_positive_example=X.shape[0]
num_positive_test=X_test.shape[0]
Y_test=np.array([1 for i in range(0,X_test.shape[0])])


print "Positive exm ",num_positive_example
folder='./neg_exm/'#'./dataset/non_signs/'
x,x_test=images_and_hog(folder)
x=x[np.random.choice(x.shape[0],size=1100,replace=False),:]
X=np.append(X,x)
X_test=np.append(X_test,x_test)
X=np.reshape(X,(-1,1764))
X_test=np.reshape(X_test,(-1,1764))
print X.shape[0]
Y=np.append(Y,[0 for i in range(0,X.shape[0]-num_positive_example)])
Y=np.reshape(Y,(-1,1))
Y_test=np.append(Y_test,[0 for i in range(0,X_test.shape[0]-num_positive_test)])
Y_test=np.reshape(Y_test,(-1,1))
# X=np.append(X,X_test)
# Y=np.append(Y,Y_test)
Y=np.reshape(Y,(-1,1))
X=np.reshape(X,(-1,1764))
print "shape =",X.shape,Y.shape

print X,Y



X_=tf.placeholder(tf.float32,[None,1764])
Y_=tf.placeholder(tf.float32,[None,1])
W1=tf.Variable(tf.random_normal([1764,2000],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[2000]))
W2=tf.Variable(tf.random_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[2000]))
W3=tf.Variable(tf.random_normal([2000,1],stddev=0.1))
b3=tf.Variable(tf.constant(0.1))

layer_1=tf.add(tf.matmul(X_,W1),b1)
layer_1=tf.nn.relu(layer_1)
layer_2=tf.add(tf.matmul(layer_1,W2),b2)
layer_2=tf.nn.relu(layer_2)
out=tf.add(tf.matmul(layer_2,W3),b3)
out=tf.nn.sigmoid(out)

def f1(): return tf.constant(1.0)
def f2(): return tf.constant(0.0)

cost=tf.reduce_mean(-1*Y_*tf.log(out)-1*(1-Y_)*tf.log(1-out))
train_step=tf.train.AdamOptimizer(1e-6).minimize(cost)
p=tf.constant(0.5,shape=[1])
print Y_
print out
print p
correct_prediction=tf.equal(tf.where(tf.less(p,out),tf.constant(1.0,shape=[989,1]),tf.constant(0.0,shape=[989,1])),Y_)
Accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100


tf.summary.scalar("cost",cost)
# tf.scalar_summary("accuracy",Accuracy)
summary_op=tf.summary.merge_all()

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
# tf.reset_default_graph()
writer=tf.summary.FileWriter('./Graph',sess.graph)	

saver=tf.train.Saver([W1,b1,W2,b2,W3,b3])
for i in range(0,5000):
	sess.run(train_step,feed_dict={X_:X,Y_:Y})
	summary=sess.run(summary_op,feed_dict={X_:X,Y_:Y})
	writer.add_summary(summary,i)
	if(i%20==0):
		print i
	if(i%50==0):
		print(out.eval(feed_dict={X_:X_test,Y_:Y_test}),Y_test)
		print("Accuracy : ",Accuracy.eval(feed_dict={X_:X_test,Y_:Y_test}))
saver.save(sess,'./tf_model/restore.ckpt')
print("Successfully trained")


print("Final Accuracy : ",Accuracy.eval(feed_dict={X_:X_test,Y_:Y_test}))
print(out.eval(feed_dict={X_:X_test,Y_:Y_test}))


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
