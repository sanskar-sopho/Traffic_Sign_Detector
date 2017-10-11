import cv2
import numpy as np
import sys
	
count=62
def patch(event,x,y,flags,param):
	if event==cv2.EVENT_LBUTTONDOWN:
		global count
		print x,y
		img2=np.zeros((64,64,3),dtype=np.uint8)
		for i in range(y,y+64):
			for j in range(x,x+64):
				img2[i-y,j-x]=img[i,j]
		cv2.imshow('patch',img2)
		cv2.imwrite('dataset/non_signs/'+str(count)+'.jpg',img2)
		count+=1
		# cv2.waitKey(0)


img=cv2.imread(sys.argv[1])
cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.setMouseCallback('image',patch)
cv2.waitKey(0)
